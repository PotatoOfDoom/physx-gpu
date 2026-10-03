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

// pxgSlangCapsuleHfCore.slang -- `capsuleHeightfieldNarrowphaseCore`
// (convexHeightfield.cu:636-789) on the engine BDA ABI.  P9.52b-4.
//
// The CAPSULE arm of `sphereHeightfieldNarrowphase`, and the last piece the
// sphere-vs-heightfield chain was missing.  It composes the four modules the
// preceding tranches verified one at a time -- `pxgSlangHeightfield` for the
// triangle, `pxgSlangHfTriFlags` for the edge classification and `selectNormal`,
// `pxgSlangSegTriangle` for the distance, `pxgSlangCapTriangle` for the two
// contact generators -- and adds only the prologue, the patch-normal choice and
// the record tail.
//
// ===========================================================================
// UPSTREAM CANDIDATE 50 IS TRANSCRIBED, NOT REPAIRED
// ===========================================================================
//
// convexHeightfield.cu:679-684 reads:
//
//     const PxVec3 tmp = capsuleTransformCached.transform.q.getBasisVector0()
//                        * capsuleHalfHeight;
//     const PxVec3 capsuleCenterInMesh =
//         trimeshTransformCached.transform.transformInv(capsuleTransformCached.transform.p);
//     const PxVec3 capsuleDirInMesh =
//         trimeshTransformCached.transform.rotateInv(tmp);      // computed ...
//     const PxVec3 p0 = capsuleCenterInMesh + tmp;              // ... and NOT used
//     const PxVec3 p1 = capsuleCenterInMesh - tmp;
//
// `capsuleCenterInMesh` is in the HEIGHTFIELD's frame and `tmp` is in the WORLD's,
// so the segment endpoints mix a mesh-space point with a world-space direction,
// and the vector computed for the purpose -- `capsuleDirInMesh` -- is discarded.
//
// The trimesh twin proves the intent: convexMesh.cu:446-451 is the same passage
// and builds p0/p1 from `capsuleDirInMesh`.  The heightfield copy lost the
// rotation.  With an axis-aligned heightfield actor -- the ordinary case for
// static terrain -- `rotateInv` is the identity and nothing shows; with a rotated
// one the capsule's axis is wrong by that rotation, in `sqDist` and in every
// contact point.
//
// This port writes `tmp`, because a port that silently improves its source stops
// being a port -- the lesson P9.53 spent a tranche relearning.  The fixture drives
// a ROTATED heightfield precisely so the transcription is pinned rather than
// accidentally right.
//
// ===========================================================================
// `getBasisVector0()` IS NOT `rotate(q, (1,0,0))`
// ===========================================================================
//
// PxQuat::getBasisVector0 (PxQuat.h:260-265) is its own formula:
//
//     x2 = x*2;  w2 = w*2;
//     ( (w*w2) - 1 + x*x2,   (z*w2) + y*x2,   (-y*w2) + z*x2 )
//
// The y and z components agree with the generic rotate term for term (float
// addition is commutative, and each product is spelled the same way).  THE X
// COMPONENT DOES NOT: `2w^2 - 1 + 2x^2` against the generic `(w^2 - u.u) + 2x^2`.
// The two are equal only because a unit quaternion has w^2 + x^2 + y^2 + z^2 = 1,
// and they are not the same float expression.  Transcribed as the .cu spells it.
//
// ===========================================================================
// THE TWO ARMS OF THIS KERNEL DEFER DIFFERENTLY, AND ONE OF THE TWO CHANNELS IS
// DEAD -- UPSTREAM CANDIDATE 52
// ===========================================================================
//
// The SPHERE arm turns `delayContacts` into three things: a second-pass queue
// entry (`outMask = globalThreadIndex | mask`), bit 31 of the ordered word, and
// bit 31 of the NI `index`.  This CAPSULE arm has no `outMask` out-parameter at
// all -- it never queues a second-pass pair -- and it writes the NI index WITHOUT
// the deferred mask (:786 is `(nbOutputContacts << NbContactsShift) + triangleIdx`
// where the sphere arm's :556 is `delayContactMask + ...`).  Only the ordered
// word's bit 31 carries `deferred` here.
//
// The NI difference is OUTPUT-EQUIVALENT, and that is provable rather than
// hopeful: `ConvexTriNormalAndIndex::isDeferred` (convexNpCommon.h:100) is
// DEFINED AND NEVER CALLED -- a grep over the whole engine finds the accessor, the
// four sites that SET the bit (convexMesh.cu:387 and :609, convexTriangle.cuh:1876,
// convexHeightfield.cu:551) and no site that reads it.  `getNbContacts` even masks
// the bit off before shifting (`(index & ~DeferredContactMask) >> NbContactsShift`),
// so a set bit cannot corrupt the count either.  The sphere arm's mask is a dead
// write; the capsule arm's omission of it costs nothing.
//
// The SECOND-PASS QUEUE is a different matter -- that channel is live for spheres
// and absent for capsules -- but that is upstream's design as written, and this
// module reproduces it.  Both are recorded; neither is reported.
//
// The NI index is also spelled `+ triangleIdx` rather than `| triangleIdx`, and the
// two are the SAME OPERATION here: `nbOutputContacts` is at most 2 and occupies bits
// 27-28, the triangle index bits 0-26, and over disjoint bit fields addition and or
// agree.  The equivalence has a precondition -- `triangleIdx < 2^27`, i.e. fewer than
// 134 million triangles -- and past it the two part company, addition carrying into
// the count field and or truncating.  Stated because the sweep's one survivor is a
// mutant that swaps them, and a later reader should know the equivalence is bounded
// rather than accidental.
//
// ===========================================================================
// UPSTREAM CANDIDATE 53: THE "EDGE NORMAL" IS A NORMALIZED ROUNDING RESIDUE
// ===========================================================================
//
// The third arm below is the only one that sets `deferred` and the only one that
// produces a normal other than `triNormal`.  It runs ONLY when `sqDist == 0.0f`, and
// `distanceSegmentTriangleSquared` returns exactly zero on three paths that all mean
// the same thing:
//
//   * the crossing path (capsuleTriangle.cuh:196-205) -- `t` is the crossing
//     parameter and `(u,v)` the barycentrics of THAT SAME crossing point;
//   * `con0 && con1` with `sqDist3 == 0`, which requires `dist3 == 0`, i.e. p0
//     exactly in the plane, whose projection is itself;
//   * `sqDistPE == 0`, a segment exactly touching an edge.
//
// In every one of them `pointOnSegment` and `pointOnTriangle` are THE SAME POINT
// WRITTEN TWO WAYS.  Their difference is identically zero in exact arithmetic, so
// what `getNormalized()` receives is nothing but the floating-point residue -- and
// what comes out is a unit vector in an essentially arbitrary direction, or the zero
// vector where the residue cancels exactly.
//
// This is measured, not argued: the fixture prints the patch normal of every case,
// and the deferred ones produce (-1,0,0), (0,0,-1), (0,-1,0), (0.9884,-0.1521,0) and
// (0,0,0).  Three of those are PERPENDICULAR to triNormal, which makes
// `generateContacts`'s `ideom` zero, `ipt` PX_MAX_F32, and the face generator silent
// -- so on those inputs every contact comes from the edge generator instead.
//
// The comment beside the .cu line says `//edge normal`.  It describes something the
// code does not do.  Transcribed as written, recorded, not reported.

module pxgSlangCapsuleHfCore;

import pxgSlangCvxTriSetup;      // CtShape / CtContactManagerInput / the quat helpers
import pxgSlangHeightfield;      // hfGetTriangleAdj
import pxgSlangHfTriFlags;       // htfTriangleFlags / htfSelectNormal
import pxgSlangSegTriangle;      // stSegmentTriangleSquared
import pxgSlangCapTriangle;      // capGenerateContacts / capGenerateEEContacts
import pxgSlangPointTriangle;    // ptDot / ptCross / ptNormalized

// PX_MAX_F32 -- what `separation` is seeded with (:704).
static const float CH_MAX_F32 = 3.4028234663852886e+38f;
// ConvexTriNormalAndIndex (convexNpCommon.h:85-86), via the SphereTri typedef.
public static const uint CH_NB_CONTACTS_SHIFT = 27u;
// The `(nbOutputContacts && !deferred)` stamp on the ordered index (:781-782).
static const uint CH_DEFER_MASK = 0x80000000u;

// PxQuat::getBasisVector0 (PxQuat.h:260-265) -- see the header.  NOT a rotate.
public float3 chBasisVector0(float4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return float3((q.w * w2) - 1.0f + q.x * x2,
                  (q.z * w2) + q.y * x2,
                  (-q.y * w2) + q.z * x2);
}

// PxMath.h:91-94 -- `PxMin(a, b)` is `a < b ? a : b`, which pins the NaN answer to
// `b`.  The separation reduction at :779 uses it.
public float chMin(float a, float b) { return a < b ? a : b; }

public struct CapHfOut
{
    public uint nbOutputContacts;
    public float3 contact[2];
    public float sep[2];
    public uint pairOffset;
}

// `ni`, `maxDepth`, `interm` and `ordered` are the ALREADY-RESOLVED bases the CUDA
// reaches through `*sphereTriNIPtr` and friends.
public CapHfOut chCapsuleCore(uint64_t cmInputs, uint64_t transformCache,
                              uint64_t contactDistance, uint64_t gpuShapes,
                              uint64_t sphereTrimeshPair,
                              uint64_t ni, uint64_t maxDepth, uint64_t interm,
                              uint64_t ordered,
                              uint4 curPair)
{
    const uint cmIdx = curPair.x;
    const uint triangleIdx = curPair.y;
    const uint testOffset = curPair.z;

    const CtContactManagerInput cin = Ptr<CtContactManagerInput>(cmInputs)[cmIdx];

    // NO SWAP, as in the sphere arm: slot 0 is the capsule and slot 1 the
    // heightfield, unconditionally (:647).
    Ptr<CtShape> shapes = Ptr<CtShape>(gpuShapes);
    const CtShape capsuleShape = shapes[cin.shapeRef0];
    const CtShape heightfieldShape = shapes[cin.shapeRef1];

    // PxsCachedTransform, 32 B: PxQuat q @0, PxVec3 p @16.
    const uint64_t xf0 = transformCache + uint64_t(cin.transformCacheRef0) * 32UL;
    const uint64_t xf1 = transformCache + uint64_t(cin.transformCacheRef1) * 32UL;
    const float4 capQ = Ptr<float4>(xf0)[0];
    const float3 capP = float3(Ptr<float>(xf0 + 16UL)[0], Ptr<float>(xf0 + 20UL)[0],
                               Ptr<float>(xf0 + 24UL)[0]);
    const float4 meshQ = Ptr<float4>(xf1)[0];
    const float3 meshP = float3(Ptr<float>(xf1 + 16UL)[0], Ptr<float>(xf1 + 20UL)[0],
                                Ptr<float>(xf1 + 24UL)[0]);

    const uint64_t meshPtr = uint64_t(heightfieldShape.hullOrMeshOffset.x)
                           | (uint64_t(heightfieldShape.hullOrMeshOffset.y) << 32);
    const uint rows = Ptr<uint>(meshPtr)[0];
    const uint columns = Ptr<uint>(meshPtr + 4UL)[0];
    const uint64_t samples = meshPtr + 8UL;

    // The `flags` read at :656-658 is upstream candidate 48 and is not
    // transcribable: it reinterprets a POINTER VARIABLE as a PxU16.  Its only
    // consumer is `boundaryCollisions`, which is constantly true -- see
    // pxgSlangHfTriFlags, which carries the derivation and the constant.

    const float3 hfScale = float3(heightfieldShape.scaleX, heightfieldShape.scaleY,
                                  heightfieldShape.scaleZ);

    const HfTriangle t = hfGetTriangleAdj(samples, triangleIdx, hfScale, rows, columns);

    // ConvexMeshPair, 56 B: startIndex @28, roundedStartIndex @40.  The trailing
    // numbers in that declaration are RUNNING END offsets, not field offsets.
    const uint64_t pairAddr = sphereTrimeshPair + uint64_t(cmIdx) * 56UL;
    const uint pairOffset = Ptr<uint>(pairAddr + 28UL)[0] + testOffset;
    const uint pairOffsetPadded = Ptr<uint>(pairAddr + 40UL)[0] + testOffset;

    const float contactDist = Ptr<float>(contactDistance)[cin.transformCacheRef0]
                            + Ptr<float>(contactDistance)[cin.transformCacheRef1];

    // A CAPSULE's PxgShape scale is (halfHeight, radius, radius)
    // (PxgNarrowphaseCore.cpp:7515), so the radius is `.y` and the half height
    // `.x` -- the opposite way round from the sphere arm, which reads `.x` for its
    // radius because a sphere's three components are equal.
    const float capsuleRadius = capsuleShape.scaleY;
    const float capsuleHalfHeight = capsuleShape.scaleX;

    // PxTransform::transformInv(PxTransform) (PxTransform.h:162-169):
    // `qinv = q.getConjugate(); (qinv.rotate(src.p - p), qinv * src.q)`.
    // P9.94d: spelled exactly that way -- rotate by the CONJUGATE, not rotateInv
    // by q.  The two agree in every nonzero bit (each negation is exact) but NOT
    // in the sign of an exact zero: the cross term `(y*vz - z*vy)` cancels to +0
    // in both, and is then ADDED in one and SUBTRACTED in the other, so a
    // component of `meshP - capP` that is +-0 can come out with the opposite sign.
    // It reaches the stored contact through `rotate(toCapQ, pt) + toCapP` whenever
    // the rotated point is -0 there (capsuleHfCore_p94dToCapPZeroSign).
    const float4 capQinv = ctQuatConjugate(capQ);
    const float4 toCapQ = ctQuatMul(capQinv, meshQ);
    const float3 toCapP = ctQuatRotate(capQinv, meshP - capP);

    // WORLD space.  See upstream candidate 50 in the header.
    const float3 tmp = chBasisVector0(capQ) * capsuleHalfHeight;
    const float3 capsuleCenterInMesh = ctQuatRotateInv(meshQ, capP - meshP);
    // `capsuleDirInMesh = ctQuatRotateInv(meshQ, tmp)` is what the .cu computes
    // here and never uses.  It is NOT computed at all in this port: transcribing a
    // dead store would only invite a later reader to wire it up.

    const float3 p0 = capsuleCenterInMesh + tmp;
    const float3 p1 = capsuleCenterInMesh - tmp;

    const float inflatedRadius = capsuleRadius + contactDist;

    // PxgTriangle::denormalizedNormal() then getNormalized -- the same spelling as
    // the sphere arm, and NOT the rotate-then-normalize the convex path uses.
    const float3 triNormal = ptNormalized(ptCross(t.v1 - t.v0, t.v2 - t.v0));

    // Measured from the CENTRE, not from an endpoint (:693).
    const float dist = ptDot(capsuleCenterInMesh - t.v0, triNormal);

    float tSeg, u, v;
    const float sqDist = stSegmentTriangleSquared(p0, p1, t.v0, t.v1, t.v2, tSeg, u, v);

    const float sqInflatedRadius = inflatedRadius * inflatedRadius;
    float separation = CH_MAX_F32;
    bool deferred = false;
    uint nbOutputContacts = 0u;
    float3 patchNormal = float3(0.0f, 0.0f, 0.0f);
    CapContacts cc = capContactsInit();

    // NO `generateContact` term here: the sphere arm's gate has three conditions,
    // this one has two (:707).
    if (dist >= 0.0f && sqInflatedRadius > sqDist)
    {
        const uint triFlags = htfTriangleFlags(samples, t.adj, t.v0, t.v1, t.v2,
                                               triNormal, hfScale, columns);

        if (htfSelectNormal(u, v, triFlags))
        {
            patchNormal = triNormal;
        }
        else
        {
            if (sqDist > 0.0f)
            {
                // The segment misses the triangle but no active edge claims the
                // feature: still the face normal.
                patchNormal = triNormal;
            }
            else
            {
                // sqDist == 0 -- the segment INTERSECTS the triangle -- and an
                // active edge owns the closest feature, so the normal comes from
                // the segment-to-triangle direction instead.  This is the ONLY
                // branch that sets `deferred`.
                const float3 pq = p1 - p0;
                const float3 pointOnSegment = p0 + pq * tSeg;
                const float w = 1.0f - (u + v);
                const float3 pointOnTriangle = t.v0 * w + t.v1 * u + t.v2 * v;
                patchNormal = ptNormalized(pointOnSegment - pointOnTriangle);
                deferred = true;
            }
        }

        // The FACE generator takes the UNSQUARED radius and the EDGE generator the
        // SQUARED one -- see pxgSlangCapTriangle's header.
        cc = capGenerateContacts(t.v0, t.v1, t.v2, triNormal, patchNormal,
                                 p0, p1, inflatedRadius);
        cc = capGenerateEEContacts(t.v0, t.v1, t.v2, patchNormal, p0, p1,
                                   sqInflatedRadius, cc);
        nbOutputContacts = cc.n;

        // The contacts go back into the CAPSULE's frame; the separations are
        // reduced with PxMin over the UNtransformed values (:776-781).
        for (uint i = 0u; i < 2u; ++i)
        {
            if (i < nbOutputContacts)
            {
                cc.pt[i] = ctQuatRotate(toCapQ, cc.pt[i]) + toCapP;
                separation = chMin(separation, cc.sep[i]);
            }
        }
    }

    Ptr<uint>(interm)[pairOffset] = triangleIdx;
    // OR here, and the sphere arm's tail uses OR too -- but note the sphere arm's
    // NI index uses ADD.  Both are transcribed as written.
    const uint deferMask = (nbOutputContacts != 0u && !deferred) ? CH_DEFER_MASK : 0u;
    Ptr<uint>(ordered)[pairOffsetPadded] = deferMask | triangleIdx;

    Ptr<float>(maxDepth)[pairOffset] = separation;

    // Rotation ALONE, by the mesh-to-capsule transform (:785).  The sphere arm
    // spells the same thing as two rotations through the world; this one has the
    // composed transform already.
    const float3 niNormal = ctQuatRotate(toCapQ, -patchNormal);
    // NO DEFERRED MASK -- see upstream candidate 52 in the header.
    const uint niIndex = (nbOutputContacts << CH_NB_CONTACTS_SHIFT) + triangleIdx;

    Ptr<uint>(ni)[4u * pairOffset + 0u] = asuint(niNormal.x);
    Ptr<uint>(ni)[4u * pairOffset + 1u] = asuint(niNormal.y);
    Ptr<uint>(ni)[4u * pairOffset + 2u] = asuint(niNormal.z);
    Ptr<uint>(ni)[4u * pairOffset + 3u] = niIndex;

    CapHfOut o;
    o.nbOutputContacts = nbOutputContacts;
    o.contact[0] = cc.pt[0];
    o.contact[1] = cc.pt[1];
    o.sep[0] = cc.sep[0];
    o.sep[1] = cc.sep[1];
    o.pairOffset = pairOffset;
    return o;
}
