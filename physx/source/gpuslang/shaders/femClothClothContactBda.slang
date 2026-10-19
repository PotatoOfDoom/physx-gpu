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

// femClothClothContactBda.slang -- the cloth-vs-CLOTH contact chain (self-collision and
// cloth-vs-different-cloth), PhysX 5.12:
//
//   cloth_clothContactPrepareLaunch (FEMClothConstraintPrep.cu)  clamps the contact count
//       to maxContacts and, for a SELF pair only, applies the exact rest-pose filter the
//       midphase approximated: the pair is INVALIDATED when the rest distance (closestPtEdge-
//       Edge for an edge-edge pair, Gu::closestPtPointTriangle2 for vertex-triangle) is
//       within mSelfCollisionFilterDistance.  A pair between two cloths stands as
//       generated.  (5.6.1 marked passing pairs VALID instead; in 5.12 the pairs arrive
//       valid.)  The kernel runs once for the VT stream and once for the EE stream; the
//       ref0 feature type (eTRIANGLE_EDGE) tells them apart.
//   cloth_queryClothClothContact{VT,EE}CountLaunch (FEMClothExternalSolve.cu)  the per-
//       substep pre-count: recomputes the contact geometry (computeClothClothVTContact /
//       EEContact), latches the result in the pair's ACTIVE flag and, for an active pair,
//       adds |weight| (> 1e-3) to mDeltaPos.w of each of the four vertices (the magnitude-
//       weighted reference count, PX_CLOTH_CLOTH_INTEGER_REFCOUNT == 0).
//   cloth_solveClothClothDelta{VT,EE}Launch (FEMClothExternalSolve.cu)  for each active
//       pair: recompute the geometry, fill two PxgDeformablePart by hand (vertex + triangle
//       for VT, edge + edge for EE; vertexInvMasses scaled by the .w counts), solveDbDbContact
//       with the 0.25 * sum of the four vertices' mDynamicFrictions, and scatter +-delta
//       (writeCloth for VT, writeClothEdge for EE).  Position iterations take the motion
//       since the last narrowphase refresh (mPrevPositionInContactOffset) as linDelta and
//       posDelta and the de-biased initPen (computeClothClothInitPen); velocity iterations
//       take velocity * dt and the current separation.
//   cloth_updateClothContactValidityLaunch (FEMClothExternalSolve.cu, unchanged in 5.12)
//       on an adaptive pair update: resets the VT/EE counts and snapshots the positions
//       into mPrevPositionInContactOffset (w = 1).
//
// THE 5.12 DATA.  The contacts are PxgDeformablePairInfo {ref0, ref1}: VT is ref0 = a cloth
// VERTEX, ref1 = a cloth TRIANGLE; EE is ref0 / ref1 = eTRIANGLE_EDGE refs (triangle +
// localEdge).  The 5.6.1 PxgFemFemContactInfo with its auxInd bit field is gone: validity is
// the INVALID flag, in-collision the ACTIVE flag, both on ref1.  The cloth friction and
// restitution are not read from the materials (clothMaterials is unread, as in the CUDA).
//
// DEVIATIONS (value-neutral):
//   - mDeltaPos accumulation uses real float atomics (pxgSlangAtomicFloat via the utils),
//     as the CUDA atomicAdd;
//   - the flag writes rewrite the ref1 word only (the CUDA rewrites ref1 in place too);
//   - clothMaterials (prepare), forceUpdateClothContactPairs and dt (updateValidity) are in
//     the push blob for ABI parity and unread, as in the CUDA.
//
// FLOAT SPELLING.  A vector divided by a scalar is one reciprocal and three multiplies
// (PxVec3::operator/); PxMin / PxMax / PxClamp on floats are the dp* selects; PxVec3 dot is
// (x + y) + z.  In PXG_NOCONTRACT_SHADERS; compared bit-exactly against a CPU transcription
// of the 5.12 CUDA (testFemClothSolve512).

import pxgSlangFEMClothDesc;                // PxgFEMClothGpu, clothAt
import pxgSlangDeformableRef;               // PxgDeformablePairInfoGpu, PxgGet*, PXG_DREF_*
import pxgSlangDeformableAndParticleUtils;  // PxgDeformablePart, solveDbDbContact, dp* helpers
import pxgSlangAtomicFloat;                 // pxgAtomicAddF4W

// PxgSoftBodyKernelBlockDim/GridDim::SB_UPDATEROTATION -- the prepare / query / solve grid
// and block; PxgFEMClothKernelBlockDim::CLOTH_STEP -- the updateContactValidity block.
static const uint RC_BLOCK = 256u;
static const uint RC_GRID = 1024u;
static const uint CLOTH_STEP = 1024u;

// FEMClothUtil.cuh.
static const float FEMCLOTH_THRESHOLD = 1.0e-14f;
static const float FEMCLOTH_NORMAL_FALLBACK_DISTSQ = 1.0e-10f;   // new in 5.12

// deformableCollision.cuh.
static const float DEFORMABLE_BARYCENTRIC_THRESHOLD = 1.0e-6f;
static const float DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD = 1.0f - 1.0e-6f;

// Index x/y/z of a uint4 by a runtime edge index in {0,1,2}
// (CUDA reinterpret_cast<const PxU32*>(&triVertInd)[localIndex]).
uint pick3(uint4 v, uint i) { return (i == 0u) ? v.x : ((i == 1u) ? v.y : v.z); }

// The pair's ref1 word, rewritten whole (PxgInvalidatePairInfo / PxgSetActivePairInfo).
void ccWriteRef1(uint64_t contactInfos, uint workIndex, uint64_t ref1)
{
    Ptr<uint64_t>(contactInfos + uint64_t(workIndex) * PXG_PAIRINFO_STRIDE + PXG_PAIRINFO_ref1)[0] = ref1;
}

// ---------------------------------------------------------------------------
// Geometry leaves (deformableCollision.cuh / GuDistancePointTriangle.h, unchanged in 5.12).
// ---------------------------------------------------------------------------

// closestPtPointTriangle (deformableCollision.cuh:146) — returns closest point, s/t out.
float3 closestPtPointTriangle(float3 p, float3 a, float3 b, float3 c, out float s, out float t)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;

    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { s = 0.0f; t = 0.0f; return a; }

    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { s = 1.0f; t = 0.0f; return b; }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        s = v; t = 0.0f;
        return a + v * ab;
    }

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { s = 0.0f; t = 1.0f; return c; }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        s = 0.0f; t = w;
        return a + w * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        s = 1.0f - w; t = w;
        return b + w * (c - b);
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    s = v; t = w;
    return a + ab * v + ac * w;
}

// Gu::closestPtPointTriangle2 (GuDistancePointTriangle.h:44) — ab / ac precomputed.  The
// EXACT rest-distance filter the prepare uses for VT self-contacts.
float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return a + ab * v + ac * w;
}

// closestPtLineLine (deformableCollision.cuh:244) — infinite-line closest points; returns
// false for a parallel / degenerate pair.  P9.36: `const PxVec3 u0 = r0 / r0Len;` (:265) is
// PxVec3::operator/, i.e. a reciprocal and three multiplies; this comment used to claim it
// was per-component division "matching the p72ref transcription", which is the convention
// corrected in the FLOAT LOWERING note above.
// UPSTREAM (transcribed as written): s = (b0 - b1 * a01) * invDet scales the u1.d term by
// 1 / r1Len where the line-line minimizer has 1 / r0Len, and t scales the u0.d term by
// 1 / r0Len where it has 1 / r1Len, so s / t (and distSq) are the true closest points only for
// two edges of equal length.
bool closestPtLineLine(float3 e0v0, float3 e0v1, float3 e1v0, float3 e1v1, out float s, out float t, out float distSq)
{
    const float3 r0 = e0v1 - e0v0;
    const float3 r1 = e1v1 - e1v0;
    const float3 d = e1v0 - e0v0;

    const float r0Len2 = dot(r0, r0);
    const float r1Len2 = dot(r1, r1);

    if (r0Len2 < 1.0e-18f || r1Len2 < 1.0e-18f)
    {
        s = 0.0f; t = 0.0f;
        distSq = dot(d, d);
        return false;
    }

    const float r0Len = sqrt(r0Len2);
    const float r1Len = sqrt(r1Len2);

    // P9.36: deformableCollision.cuh:265-266 are `const PxVec3 u0 = r0 / r0Len;` --
    // PxVec3::operator/ (PxVec3.h:219) is ONE reciprocal and three multiplies.  The two
    // scalar divides further down (b0, b1) are scalar/scalar and stay divides.
    const float3 u0 = r0 * (1.0f / r0Len);
    const float3 u1 = r1 * (1.0f / r1Len);

    const float cosTheta = abs(dot(u0, u1));

    if (cosTheta > 0.9999f)
    {
        const float tProj = dot(e0v0 - e1v0, r1) / r1Len2;
        t = tProj;
        s = 0.0f;

        const float3 pt0 = e0v0;
        const float3 pt1 = e1v0 + r1 * tProj;
        distSq = dot(pt1 - pt0, pt1 - pt0);
        return false;
    }

    const float a01 = -dot(u0, u1);
    const float b0 = dot(u0, d) / r0Len;
    const float b1 = -dot(u1, d) / r1Len;

    const float det = 1.0f - a01 * a01;

    if (det < 1.0e-9f)
    {
        s = 0.0f; t = 0.0f;
        distSq = dot(d, d);
        return false;
    }

    const float invDet = 1.0f / det;
    s = (b0 - b1 * a01) * invDet;
    t = (b1 - a01 * b0) * invDet;

    const float3 pt0 = e0v0 + s * r0;
    const float3 pt1 = e1v0 + t * r1;
    distSq = dot(pt1 - pt0, pt1 - pt0);

    return true;
}

// closestPtPointEdge (deformableCollision.cuh:231) — the closestPtEdgeEdge segment
// fallback leaf.
float closestPtPointEdge(float3 v0, float3 ev0, float3 ev1)
{
    const float3 r = ev1 - ev0;
    const float d2 = dot(r, r);
    // PxClamp(x, 0, 1) = PxMax(0, PxMin(1, x)): the CUDA selects (a +-0 tie keeps the
    // second operand), not GLSL FMin/FMax.
    return (d2 > 0.0f) ? dpSelectMax(0.0f, dpSelectMin(1.0f, dot(r, v0 - ev0) / d2)) : 0.5f;
}

// closestPtEdgeEdge (deformableCollision.cuh:313) — finite-edge closest points; the EXACT
// rest-distance filter the prepare uses for EE self-contacts.  Falls back to the point-edge
// candidates when the infinite-line solution leaves the [0,1] x [0,1] box.
bool closestPtEdgeEdge(float3 e0v0, float3 e0v1, float3 e1v0, float3 e1v1, out float s, out float t, out float distSq)
{
    if (!closestPtLineLine(e0v0, e0v1, e1v0, e1v1, s, t, distSq))
        return false;

    if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f)
    {
        float w[4];
        w[0] = closestPtPointEdge(e0v0, e1v0, e1v1);
        w[1] = closestPtPointEdge(e0v1, e1v0, e1v1);
        w[2] = closestPtPointEdge(e1v0, e0v0, e0v1);
        w[3] = closestPtPointEdge(e1v1, e0v0, e0v1);

        float2 candidates[4];
        candidates[0] = float2(0.0f, w[0]);
        candidates[1] = float2(1.0f, w[1]);
        candidates[2] = float2(w[2], 0.0f);
        candidates[3] = float2(w[3], 1.0f);

        int minIndex = 0;
        distSq = 3.4028234663852886e+38f;  // PX_MAX_REAL
        for (int i = 0; i < 4; ++i)
        {
            const float sC = candidates[i].x;
            const float tC = candidates[i].y;
            const float3 p0 = e0v0 * (1.0f - sC) + e0v1 * sC;
            const float3 p1 = e1v0 * (1.0f - tC) + e1v1 * tC;
            const float d2 = dot(p0 - p1, p0 - p1);
            minIndex = (d2 < distSq) ? i : minIndex;
            distSq = dpSelectMin(d2, distSq);   // PxMin, a select
        }

        s = candidates[minIndex].x;
        t = candidates[minIndex].y;
    }

    return true;
}

// ---------------------------------------------------------------------------
// computeClothClothVTContact / computeClothClothEEContact / computeClothClothInitPen
// (FEMClothExternalSolve.cu).  weights is (w0, w1, w2, w3), cloth0 minus cloth1.
// ---------------------------------------------------------------------------
float3 f4xyz(float4 v) { return float3(v.x, v.y, v.z); }

// disp = w0 * xx0 + w1 * xx1 + w2 * xx2 + w3 * xx3, left to right.
float3 ccWeightedSum(float4 w, float3 xx0, float3 xx1, float3 xx2, float3 xx3)
{
    return float3(xx0.x * w.x + xx1.x * w.y + xx2.x * w.z + xx3.x * w.w,
                  xx0.y * w.x + xx1.y * w.y + xx2.y * w.z + xx3.y * w.w,
                  xx0.z * w.x + xx1.z * w.y + xx2.z * w.z + xx3.z * w.w);
}

bool computeClothClothVTContact(Ptr<PxgFEMClothGpu> cloth0, Ptr<PxgFEMClothGpu> cloth1, float4 x0, float4 x1,
                                float4 x2, float4 x3, out float4 weights, out float3 normalFromDisp, out float3 disp)
{
    const float3 xx0 = f4xyz(x0);
    const float3 xx1 = f4xyz(x1);
    const float3 xx2 = f4xyz(x2);
    const float3 xx3 = f4xyz(x3);

    float s, t;   // Barycentric coordinates
    closestPtPointTriangle(xx0, xx1, xx2, xx3, s, t);
    const float r = 1.0f - s - t;
    weights = float4(1.0f, -r, -s, -t);   // cloth0 - cloth1

    disp = ccWeightedSum(weights, xx0, xx1, xx2, xx3);
    normalFromDisp = float3(0.0f, 0.0f, 0.0f);
    const float dispSq = dpDot3(disp, disp);
    const float restDist = cloth0[0].mRestOffset + cloth1[0].mRestOffset;

    if (dispSq > restDist * restDist)
        return false;

    if (dispSq < FEMCLOTH_NORMAL_FALLBACK_DISTSQ)
    {
        const float3 ab = xx2 - xx1;
        const float3 ac = xx3 - xx1;
        normalFromDisp = dpCross3(ab, ac);

        const float nLenSq = dpDot3(normalFromDisp, normalFromDisp);
        if (nLenSq < FEMCLOTH_THRESHOLD)
            return false;   // Degenerate triangle.

        const float invLen = 1.0f / sqrt(nLenSq);
        normalFromDisp = float3(normalFromDisp.x * invLen, normalFromDisp.y * invLen, normalFromDisp.z * invLen);

        const float3 toVertex = xx0 - xx1;
        if (dpDot3(normalFromDisp, toVertex) < 0.0f)
            normalFromDisp = float3(-normalFromDisp.x, -normalFromDisp.y, -normalFromDisp.z);
    }
    else
    {
        const float invLen = 1.0f / sqrt(dispSq);
        normalFromDisp = float3(disp.x * invLen, disp.y * invLen, disp.z * invLen);
    }

    return true;
}

bool computeClothClothEEContact(Ptr<PxgFEMClothGpu> cloth0, Ptr<PxgFEMClothGpu> cloth1, float4 x0, float4 x1,
                                float4 x2, float4 x3, out float4 weights, out float3 normalFromDisp, out float3 disp)
{
    const float3 xx0 = f4xyz(x0);
    const float3 xx1 = f4xyz(x1);
    const float3 xx2 = f4xyz(x2);
    const float3 xx3 = f4xyz(x3);

    weights = float4(0.0f, 0.0f, 0.0f, 0.0f);
    normalFromDisp = float3(0.0f, 0.0f, 0.0f);
    disp = float3(0.0f, 0.0f, 0.0f);

    // Vertex-edge collisions are handled in vertex-triangle collisions.
    float s, t;
    float distSq;
    const bool isValid = closestPtLineLine(xx0, xx1, xx2, xx3, s, t, distSq);
    const bool isOutside = ((s < DEFORMABLE_BARYCENTRIC_THRESHOLD) || (s > DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD) ||
                            (t < DEFORMABLE_BARYCENTRIC_THRESHOLD) || (t > DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD));

    if (isOutside)
        return false;

    weights = float4(1.0f - s, s, -(1.0f - t), -t);   // cloth0 - cloth1
    disp = ccWeightedSum(weights, xx0, xx1, xx2, xx3);
    const float dispSq = dpDot3(disp, disp);
    const float restDist = cloth0[0].mRestOffset + cloth1[0].mRestOffset;

    if (dispSq > restDist * restDist)
        return false;

    // The CUDA tests and normalizes by the closestPtLineLine distSq here, not dispSq.
    if (distSq < FEMCLOTH_NORMAL_FALLBACK_DISTSQ)
    {
        if (!isValid)
            return false;   // Parallel / zero-length AND close: normal undefined.
        normalFromDisp = dpCross3(xx1 - xx0, xx3 - xx2);
        const float nLenSq = dpDot3(normalFromDisp, normalFromDisp);
        if (nLenSq < FEMCLOTH_THRESHOLD)
            return false;
        const float invLen = 1.0f / sqrt(nLenSq);
        normalFromDisp = float3(normalFromDisp.x * invLen, normalFromDisp.y * invLen, normalFromDisp.z * invLen);
    }
    else
    {
        const float invLen = 1.0f / sqrt(distSq);
        normalFromDisp = float3(disp.x * invLen, disp.y * invLen, disp.z * invLen);
    }

    if (!isValid)
    {
        // Degenerate cases: re-adjust the weights so the constraint applies at the edge midpoints.
        weights = float4(0.5f, 0.5f, -0.5f, -0.5f);
    }

    return true;
}

// (disp - relDelta) . normalFromDisp - restDist, relDelta the weighted motion since prevX.
float computeClothClothInitPen(Ptr<PxgFEMClothGpu> cloth0, Ptr<PxgFEMClothGpu> cloth1, float4 x0, float4 x1,
                               float4 x2, float4 x3, float4 prevX0, float4 prevX1, float4 prevX2, float4 prevX3,
                               float4 weights, float3 normalFromDisp, float3 disp)
{
    const float4 relDelta4 = weights.x * (x0 - prevX0) + weights.y * (x1 - prevX1) +
                             weights.z * (x2 - prevX2) + weights.w * (x3 - prevX3);
    const float3 relDelta = float3(relDelta4.x, relDelta4.y, relDelta4.z);
    const float restDist = cloth0[0].mRestOffset + cloth1[0].mRestOffset;
    return dpDot3(disp - relDelta, normalFromDisp) - restDist;
}

// The four EE vertex indices: edge0 of triangle ref0, edge1 of triangle ref1.
uint4 ccEdgeEdgeVertices(Ptr<PxgFEMClothGpu> cloth0, Ptr<PxgFEMClothGpu> cloth1, uint64_t ref0, uint64_t ref1)
{
    const uint e0_localIndex0 = PxgGetDeformableEdgeLocalIndex(ref0);
    const uint e1_localIndex0 = PxgGetDeformableEdgeLocalIndex(ref1);
    const uint e0_localIndex1 = (e0_localIndex0 + 1u) % 3u;
    const uint e1_localIndex1 = (e1_localIndex0 + 1u) % 3u;
    const uint4 triVertInd0 = Ptr<uint4>(cloth0[0].mTriangleVertexIndices_ownedEdges)[PxgGetDeformableEdgeTriangle(ref0)];
    const uint4 triVertInd1 = Ptr<uint4>(cloth1[0].mTriangleVertexIndices_ownedEdges)[PxgGetDeformableEdgeTriangle(ref1)];
    return uint4(pick3(triVertInd0, e0_localIndex0), pick3(triVertInd0, e0_localIndex1),
                 pick3(triVertInd1, e1_localIndex0), pick3(triVertInd1, e1_localIndex1));
}

// ===========================================================================
// Push blobs.
// ===========================================================================
// cloth_clothContactPrepareLaunch(PxgFEMCloth* clothes, PxgDeformablePairInfo* contactInfos,
//   PxU32* numContacts, PxU32 maxContacts, PxsDeformableSurfaceMaterialData* clothMaterials,
//   const PxU8* updateContactPairs) -> {8, 8, 8, 4, 8, 8} = 48-byte blob.
struct PrepClothClothParams
{
    uint64_t clothes;             // @0
    uint64_t contactInfos;        // @8   in/out (INVALID flag)
    uint64_t numContacts;         // @16  in/out (clamped)
    uint maxContacts;             // @24
    uint _pad28;                  // @28
    uint64_t clothMaterials;      // @32  (unread, as in CUDA)
    uint64_t updateContactPairs;  // @40  -> 48
}

// cloth_queryClothClothContact{VT,EE}CountLaunch(PxgFEMCloth* cloths,
//   PxgDeformablePairInfo* contactInfos, PxU32* numContacts) -> {8, 8, 8} = 24-byte blob.
struct QueryClothClothParams
{
    uint64_t cloths;         // @0
    uint64_t contactInfos;   // @8   in/out (ACTIVE flag)
    uint64_t numContacts;    // @16 -> 24
}

// cloth_solveClothClothDelta{VT,EE}Launch(PxgFEMCloth* cloths,
//   const PxgDeformablePairInfo* contactInfos, PxU32* numContacts, PxReal dt,
//   bool isVelocityIteration) -> {8, 8, 8, 4, 1} = 32-byte blob.
struct SolveClothClothParams
{
    uint64_t cloths;              // @0
    uint64_t contactInfos;        // @8
    uint64_t numContacts;         // @16
    float dt;                     // @24
    uint isVelocityIteration;     // @28  bool byte0 -> 32
}

// cloth_updateClothContactValidityLaunch(PxgFEMCloth* femCloths, const PxU32* activeIds,
//   bool adaptiveCollisionPairUpdate, bool forceUpdateClothContactPairs,
//   const PxU8* updateContactPairs, PxU32* VTContactCount, PxU32* EEContactCount, PxReal dt)
//   -> {8, 8, 1, 1, 8, 8, 8, 4} = 56-byte blob.
struct UpdateValidityParams
{
    uint64_t femCloths;                 // @0
    uint64_t activeIds;                 // @8
    uint adaptiveFlagWord;              // @16  byte0 = adaptiveCollisionPairUpdate,
                                        //      byte1 = forceUpdateClothContactPairs (unread)
    uint _pad20;                        // @20
    uint64_t updateContactPairs;        // @24
    uint64_t VTContactCount;            // @32
    uint64_t EEContactCount;            // @40
    float dt;                           // @48  (unread, as in CUDA)
    uint _pad52;                        // @52 -> 56
}

[[vk::push_constant]] ConstantBuffer<PrepClothClothParams> gPrep;
[[vk::push_constant]] ConstantBuffer<QueryClothClothParams> gQuery;
[[vk::push_constant]] ConstantBuffer<SolveClothClothParams> gSolve;
[[vk::push_constant]] ConstantBuffer<UpdateValidityParams> gUpdate;

// ===========================================================================
// cloth_clothContactPrepareLaunch
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void cloth_clothContactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    // Early exit if contact pairs are not updated (the PxU8 gate, read as its word's low byte).
    if ((Ptr<uint>(gPrep.updateContactPairs)[0] & 0xffu) == 0u)
        return;

    const uint tNumContacts = min(Ptr<uint>(gPrep.numContacts)[0], gPrep.maxContacts);
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;

        if (workIndex == 0u)   // Clamp the cloth contact count.
            Ptr<uint>(gPrep.numContacts)[0] = tNumContacts;

        if (workIndex >= tNumContacts)
            return;

        const PxgDeformablePairInfoGpu contactInfo = PxgLoadPairInfo(gPrep.contactInfos, workIndex);

        const uint clothId0 = PxgGetDeformableId(contactInfo.ref0);
        const uint clothId1 = PxgGetDeformableId(contactInfo.ref1);

        // Only self-collision needs the exact test.  A contact between two cloths stands as
        // generated, the midphase having applied the filter distance that self-collision only
        // approximates.
        if (clothId0 != clothId1)
            continue;

        Ptr<PxgFEMClothGpu> cloth0 = clothAt(gPrep.clothes, clothId0);
        Ptr<PxgFEMClothGpu> cloth1 = clothAt(gPrep.clothes, clothId1);
        const float filterDistance = cloth0[0].mSelfCollisionFilterDistance;

        if (PxgGetDeformableFeatureType(contactInfo.ref0) == PXG_DREF_eTRIANGLE_EDGE)
        {
            const uint4 vertIndices = ccEdgeEdgeVertices(cloth0, cloth1, contactInfo.ref0, contactInfo.ref1);

            // Compute the exact rest distance for filtering.
            const float3 r0 = Ptr<float4>(cloth0[0].mRestPosition)[vertIndices.x].xyz;
            const float3 r1 = Ptr<float4>(cloth0[0].mRestPosition)[vertIndices.y].xyz;
            const float3 r2 = Ptr<float4>(cloth1[0].mRestPosition)[vertIndices.z].xyz;
            const float3 r3 = Ptr<float4>(cloth1[0].mRestPosition)[vertIndices.w].xyz;

            float s, t;
            float restDistSq;
            closestPtEdgeEdge(r0, r1, r2, r3, s, t, restDistSq);

            if (restDistSq <= filterDistance * filterDistance)
                ccWriteRef1(gPrep.contactInfos, workIndex, PxgInvalidatePairInfo(contactInfo).ref1);
        }
        else   // Vertex-triangle collision
        {
            const uint4 triVertId1 = Ptr<uint4>(cloth1[0].mTriangleVertexIndices_ownedEdges)[PxgGetDeformableFeatureId(contactInfo.ref1)];

            const float3 r0 = Ptr<float4>(cloth0[0].mRestPosition)[PxgGetDeformableFeatureId(contactInfo.ref0)].xyz;
            const float3 r1 = Ptr<float4>(cloth1[0].mRestPosition)[triVertId1.x].xyz;
            const float3 r2 = Ptr<float4>(cloth1[0].mRestPosition)[triVertId1.y].xyz;
            const float3 r3 = Ptr<float4>(cloth1[0].mRestPosition)[triVertId1.z].xyz;

            const float3 r12 = r2 - r1;
            const float3 r13 = r3 - r1;

            const float3 closest = closestPtPointTriangle2(r0, r1, r2, r3, r12, r13);

            const float3 rd = r0 - closest;
            const float restDistSq = dpDot3(rd, rd);
            if (restDistSq <= filterDistance * filterDistance)
                ccWriteRef1(gPrep.contactInfos, workIndex, PxgInvalidatePairInfo(contactInfo).ref1);
        }
    }
}

// ===========================================================================
// cloth_queryClothClothContact{VT,EE}CountLaunch
// ===========================================================================
// The magnitude-weighted reference-count bump (PX_CLOTH_CLOTH_INTEGER_REFCOUNT == 0).
void ccBumpAbs(uint64_t deltaPos, uint vertIdx, float w)
{
    if (abs(w) > 1e-3f)
        pxgAtomicAddF4W(deltaPos, vertIdx, abs(w));
}

[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void cloth_queryClothClothContactVTCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gQuery.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + groupThreadId.x + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const PxgDeformablePairInfoGpu contactInfo = PxgLoadPairInfo(gQuery.contactInfos, workIndex);
        if (!PxgIsValidPairInfo(contactInfo))
            continue;

        Ptr<PxgFEMClothGpu> cloth0 = clothAt(gQuery.cloths, PxgGetDeformableId(contactInfo.ref0));
        Ptr<PxgFEMClothGpu> cloth1 = clothAt(gQuery.cloths, PxgGetDeformableId(contactInfo.ref1));

        const uint elementId0 = PxgGetDeformableFeatureId(contactInfo.ref0);
        const uint elementId1 = PxgGetDeformableFeatureId(contactInfo.ref1);

        const uint4 triVertId1 = Ptr<uint4>(cloth1[0].mTriangleVertexIndices_ownedEdges)[elementId1];
        const uint4 vertIndices = uint4(elementId0, triVertId1.x, triVertId1.y, triVertId1.z);

        const float4 x0 = Ptr<float4>(cloth0[0].mPosition_InvMass)[vertIndices.x];
        const float4 x1 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.y];
        const float4 x2 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.z];
        const float4 x3 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.w];

        float4 weights;
        float3 normalFromDisp;
        float3 disp;
        if (!computeClothClothVTContact(cloth0, cloth1, x0, x1, x2, x3, weights, normalFromDisp, disp))
        {
            ccWriteRef1(gQuery.contactInfos, workIndex, PxgSetActiveRef1(contactInfo.ref1, false));
            continue;
        }

        // VT layout: vert 0 -> cloth0; verts 1..3 -> cloth1 (triangle).
        ccBumpAbs(cloth0[0].mDeltaPos, vertIndices.x, weights.x);
        ccBumpAbs(cloth1[0].mDeltaPos, vertIndices.y, weights.y);
        ccBumpAbs(cloth1[0].mDeltaPos, vertIndices.z, weights.z);
        ccBumpAbs(cloth1[0].mDeltaPos, vertIndices.w, weights.w);
        ccWriteRef1(gQuery.contactInfos, workIndex, PxgSetActiveRef1(contactInfo.ref1, true));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void cloth_queryClothClothContactEECountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gQuery.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + groupThreadId.x + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const PxgDeformablePairInfoGpu contactInfo = PxgLoadPairInfo(gQuery.contactInfos, workIndex);
        if (!PxgIsValidPairInfo(contactInfo))
            continue;

        Ptr<PxgFEMClothGpu> cloth0 = clothAt(gQuery.cloths, PxgGetDeformableId(contactInfo.ref0));
        Ptr<PxgFEMClothGpu> cloth1 = clothAt(gQuery.cloths, PxgGetDeformableId(contactInfo.ref1));

        const uint4 vertIndices = ccEdgeEdgeVertices(cloth0, cloth1, contactInfo.ref0, contactInfo.ref1);

        const float4 x0 = Ptr<float4>(cloth0[0].mPosition_InvMass)[vertIndices.x];
        const float4 x1 = Ptr<float4>(cloth0[0].mPosition_InvMass)[vertIndices.y];
        const float4 x2 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.z];
        const float4 x3 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.w];

        float4 weights;
        float3 normalFromDisp;
        float3 disp;
        if (!computeClothClothEEContact(cloth0, cloth1, x0, x1, x2, x3, weights, normalFromDisp, disp))
        {
            ccWriteRef1(gQuery.contactInfos, workIndex, PxgSetActiveRef1(contactInfo.ref1, false));
            continue;
        }

        // EE layout: verts 0..1 -> cloth0 (edge); verts 2..3 -> cloth1 (edge).
        ccBumpAbs(cloth0[0].mDeltaPos, vertIndices.x, weights.x);
        ccBumpAbs(cloth0[0].mDeltaPos, vertIndices.y, weights.y);
        ccBumpAbs(cloth1[0].mDeltaPos, vertIndices.z, weights.z);
        ccBumpAbs(cloth1[0].mDeltaPos, vertIndices.w, weights.w);
        ccWriteRef1(gQuery.contactInfos, workIndex, PxgSetActiveRef1(contactInfo.ref1, true));
    }
}

// ===========================================================================
// cloth_solveClothClothDelta{VT,EE}Launch
// ===========================================================================
[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void cloth_solveClothClothDeltaVTLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSolve.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const float dt = gSolve.dt;
    const bool isVelocityIteration = (gSolve.isVelocityIteration & 0xffu) != 0u;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + groupThreadId.x + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const PxgDeformablePairInfoGpu contactInfo = PxgLoadPairInfo(gSolve.contactInfos, workIndex);
        if (!PxgIsActivePairInfo(contactInfo))
            continue;

        Ptr<PxgFEMClothGpu> cloth0 = clothAt(gSolve.cloths, PxgGetDeformableId(contactInfo.ref0));
        Ptr<PxgFEMClothGpu> cloth1 = clothAt(gSolve.cloths, PxgGetDeformableId(contactInfo.ref1));

        const uint elementId0 = PxgGetDeformableFeatureId(contactInfo.ref0);
        const uint elementId1 = PxgGetDeformableFeatureId(contactInfo.ref1);

        const uint4 triVertId1 = Ptr<uint4>(cloth1[0].mTriangleVertexIndices_ownedEdges)[elementId1];
        const uint4 vertIndices = uint4(elementId0, triVertId1.x, triVertId1.y, triVertId1.z);

        const float4 x0 = Ptr<float4>(cloth0[0].mPosition_InvMass)[vertIndices.x];
        const float4 x1 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.y];
        const float4 x2 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.z];
        const float4 x3 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.w];

        const float frictionCoefficient = 0.25f * (Ptr<float>(cloth0[0].mDynamicFrictions)[vertIndices.x] +
                                                   Ptr<float>(cloth1[0].mDynamicFrictions)[vertIndices.y] +
                                                   Ptr<float>(cloth1[0].mDynamicFrictions)[vertIndices.z] +
                                                   Ptr<float>(cloth1[0].mDynamicFrictions)[vertIndices.w]);

        // Refcounts populated by the count pass (magnitude weighting).
        const float count0 = Ptr<float4>(cloth0[0].mDeltaPos)[vertIndices.x].w;
        const float count1 = Ptr<float4>(cloth1[0].mDeltaPos)[vertIndices.y].w;
        const float count2 = Ptr<float4>(cloth1[0].mDeltaPos)[vertIndices.z].w;
        const float count3 = Ptr<float4>(cloth1[0].mDeltaPos)[vertIndices.w].w;

        PxgDeformablePart db0 = PxgDeformablePart();
        PxgDeformablePart db1 = PxgDeformablePart();
        PxgDbContactPair contactPair = PxgDbContactPair();
        PxgDbSolveOutput solveOut = PxgDbSolveOutput();
        float4 weights;
        float3 normalFromDisp;
        float3 disp;
        if (!computeClothClothVTContact(cloth0, cloth1, x0, x1, x2, x3, weights, normalFromDisp, disp))
            continue;   // pair separated or degenerated since the count pass

        contactPair.maxPenBiasClamp = dpSelectMax(cloth0[0].mPenBiasClamp, cloth1[0].mPenBiasClamp);
        db0.ref = contactInfo.ref0;   // cloth0 side is a single vertex
        db1.ref = contactInfo.ref1;   // cloth1 side is a triangle
        db0.bc = float4(1.0f, 0.0f, 0.0f, 0.0f);
        db1.bc = float4(-weights.y, -weights.z, -weights.w, 0.0f);
        contactPair.normal = float3(-normalFromDisp.x, -normalFromDisp.y, -normalFromDisp.z);

        // Side 0: cloth0 vertex.  Side 1: cloth1 triangle.
        db0.vertexInvMasses = float4(count0 * x0.w, 0.0f, 0.0f, 0.0f);
        db1.vertexInvMasses = float4(count1 * x1.w, count2 * x2.w, count3 * x3.w, 0.0f);

        if (!isVelocityIteration)
        {
            // prevX is the baseline at the last cloth-cloth narrowphase refresh.
            const float4 prevX0 = Ptr<float4>(cloth0[0].mPrevPositionInContactOffset)[vertIndices.x];
            const float4 prevX1 = Ptr<float4>(cloth1[0].mPrevPositionInContactOffset)[vertIndices.y];
            const float4 prevX2 = Ptr<float4>(cloth1[0].mPrevPositionInContactOffset)[vertIndices.z];
            const float4 prevX3 = Ptr<float4>(cloth1[0].mPrevPositionInContactOffset)[vertIndices.w];

            db0.linDelta = float3(x0.x - prevX0.x, x0.y - prevX0.y, x0.z - prevX0.z);
            const float4 db1Delta4 = db1.bc.x * (x1 - prevX1) + db1.bc.y * (x2 - prevX2) + db1.bc.z * (x3 - prevX3);
            db1.linDelta = float3(db1Delta4.x, db1Delta4.y, db1Delta4.z);

            contactPair.initPen = computeClothClothInitPen(cloth0, cloth1, x0, x1, x2, x3, prevX0, prevX1, prevX2,
                                                           prevX3, weights, normalFromDisp, disp);

            db0.posDelta = db0.linDelta;
            db1.posDelta = db1.linDelta;
        }
        else
        {
            // linDelta holds velocity * dt; the posDeltas stay zero and initPen carries the
            // current separation.
            contactPair.initPen = dpDot3(disp, normalFromDisp) - (cloth0[0].mRestOffset + cloth1[0].mRestOffset);
            const float4 vel0 = Ptr<float4>(cloth0[0].mVelocity_InvMass)[vertIndices.x];
            db0.linDelta = float3(vel0.x * dt, vel0.y * dt, vel0.z * dt);
            const float4 vel1 = Ptr<float4>(cloth1[0].mVelocity_InvMass)[vertIndices.y];
            const float4 vel2 = Ptr<float4>(cloth1[0].mVelocity_InvMass)[vertIndices.z];
            const float4 vel3 = Ptr<float4>(cloth1[0].mVelocity_InvMass)[vertIndices.w];
            const float4 db1Vel4 = db1.bc.x * vel1 + db1.bc.y * vel2 + db1.bc.z * vel3;
            db1.linDelta = float3(db1Vel4.x * dt, db1Vel4.y * dt, db1Vel4.z * dt);
        }

        const bool hasApplied = solveDbDbContact(db0, db1, contactPair, solveOut, 0.0f, 0.0f, frictionCoefficient, dt,
                                                 /*checkOnlyActivity*/ false, isVelocityIteration);

        if (hasApplied)
        {
            float3 delta;
            solveOut.computeDelta(contactPair, delta, dt);

            // writeCloth scatters .xyz only; the count pass owns .w this iteration.
            db0.writeCloth(cloth0, delta);
            db1.writeCloth(cloth1, float3(-delta.x, -delta.y, -delta.z));
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]   // == RC_BLOCK
void cloth_solveClothClothDeltaEELaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSolve.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const float dt = gSolve.dt;
    const bool isVelocityIteration = (gSolve.isVelocityIteration & 0xffu) != 0u;

    for (uint i = 0; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + groupThreadId.x + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const PxgDeformablePairInfoGpu contactInfo = PxgLoadPairInfo(gSolve.contactInfos, workIndex);
        if (!PxgIsActivePairInfo(contactInfo))
            continue;

        Ptr<PxgFEMClothGpu> cloth0 = clothAt(gSolve.cloths, PxgGetDeformableId(contactInfo.ref0));
        Ptr<PxgFEMClothGpu> cloth1 = clothAt(gSolve.cloths, PxgGetDeformableId(contactInfo.ref1));

        const uint4 vertIndices = ccEdgeEdgeVertices(cloth0, cloth1, contactInfo.ref0, contactInfo.ref1);

        const float4 x0 = Ptr<float4>(cloth0[0].mPosition_InvMass)[vertIndices.x];
        const float4 x1 = Ptr<float4>(cloth0[0].mPosition_InvMass)[vertIndices.y];
        const float4 x2 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.z];
        const float4 x3 = Ptr<float4>(cloth1[0].mPosition_InvMass)[vertIndices.w];

        const float frictionCoefficient = 0.25f * (Ptr<float>(cloth0[0].mDynamicFrictions)[vertIndices.x] +
                                                   Ptr<float>(cloth0[0].mDynamicFrictions)[vertIndices.y] +
                                                   Ptr<float>(cloth1[0].mDynamicFrictions)[vertIndices.z] +
                                                   Ptr<float>(cloth1[0].mDynamicFrictions)[vertIndices.w]);

        const float count0 = Ptr<float4>(cloth0[0].mDeltaPos)[vertIndices.x].w;
        const float count1 = Ptr<float4>(cloth0[0].mDeltaPos)[vertIndices.y].w;
        const float count2 = Ptr<float4>(cloth1[0].mDeltaPos)[vertIndices.z].w;
        const float count3 = Ptr<float4>(cloth1[0].mDeltaPos)[vertIndices.w].w;

        PxgDeformablePart db0 = PxgDeformablePart();
        PxgDeformablePart db1 = PxgDeformablePart();
        PxgDbContactPair contactPair = PxgDbContactPair();
        PxgDbSolveOutput solveOut = PxgDbSolveOutput();
        float4 weights;
        float3 normalFromDisp;
        float3 disp;
        if (!computeClothClothEEContact(cloth0, cloth1, x0, x1, x2, x3, weights, normalFromDisp, disp))
            continue;   // pair separated or degenerated since the count pass

        contactPair.maxPenBiasClamp = dpSelectMax(cloth0[0].mPenBiasClamp, cloth1[0].mPenBiasClamp);
        // Each side is an edge, so only two weights are used.
        db0.ref = contactInfo.ref0;
        db1.ref = contactInfo.ref1;
        db0.bc = float4(weights.x, weights.y, 0.0f, 0.0f);
        db1.bc = float4(-weights.z, -weights.w, 0.0f, 0.0f);
        contactPair.normal = float3(-normalFromDisp.x, -normalFromDisp.y, -normalFromDisp.z);

        db0.vertexInvMasses = float4(count0 * x0.w, count1 * x1.w, 0.0f, 0.0f);
        db1.vertexInvMasses = float4(count2 * x2.w, count3 * x3.w, 0.0f, 0.0f);

        if (!isVelocityIteration)
        {
            const float4 prevX0 = Ptr<float4>(cloth0[0].mPrevPositionInContactOffset)[vertIndices.x];
            const float4 prevX1 = Ptr<float4>(cloth0[0].mPrevPositionInContactOffset)[vertIndices.y];
            const float4 prevX2 = Ptr<float4>(cloth1[0].mPrevPositionInContactOffset)[vertIndices.z];
            const float4 prevX3 = Ptr<float4>(cloth1[0].mPrevPositionInContactOffset)[vertIndices.w];

            const float4 db0Delta4 = db0.bc.x * (x0 - prevX0) + db0.bc.y * (x1 - prevX1);
            db0.linDelta = float3(db0Delta4.x, db0Delta4.y, db0Delta4.z);
            const float4 db1Delta4 = db1.bc.x * (x2 - prevX2) + db1.bc.y * (x3 - prevX3);
            db1.linDelta = float3(db1Delta4.x, db1Delta4.y, db1Delta4.z);

            contactPair.initPen = computeClothClothInitPen(cloth0, cloth1, x0, x1, x2, x3, prevX0, prevX1, prevX2,
                                                           prevX3, weights, normalFromDisp, disp);

            db0.posDelta = db0.linDelta;
            db1.posDelta = db1.linDelta;
        }
        else
        {
            contactPair.initPen = dpDot3(disp, normalFromDisp) - (cloth0[0].mRestOffset + cloth1[0].mRestOffset);
            const float4 vel0 = Ptr<float4>(cloth0[0].mVelocity_InvMass)[vertIndices.x];
            const float4 vel1 = Ptr<float4>(cloth0[0].mVelocity_InvMass)[vertIndices.y];
            const float4 db0Vel4 = db0.bc.x * vel0 + db0.bc.y * vel1;
            db0.linDelta = float3(db0Vel4.x * dt, db0Vel4.y * dt, db0Vel4.z * dt);
            const float4 vel2 = Ptr<float4>(cloth1[0].mVelocity_InvMass)[vertIndices.z];
            const float4 vel3 = Ptr<float4>(cloth1[0].mVelocity_InvMass)[vertIndices.w];
            const float4 db1Vel4 = db1.bc.x * vel2 + db1.bc.y * vel3;
            db1.linDelta = float3(db1Vel4.x * dt, db1Vel4.y * dt, db1Vel4.z * dt);
        }

        const bool hasApplied = solveDbDbContact(db0, db1, contactPair, solveOut, 0.0f, 0.0f, frictionCoefficient, dt,
                                                 /*checkOnlyActivity*/ false, isVelocityIteration);

        if (hasApplied)
        {
            float3 delta;
            solveOut.computeDelta(contactPair, delta, dt);

            // vertexInvMasses is pre-scaled by count, so the > 0 gate inside writeClothEdge
            // also covers count == 0.
            db0.writeClothEdge(cloth0, vertIndices.x, vertIndices.y, delta);
            db1.writeClothEdge(cloth1, vertIndices.z, vertIndices.w, float3(-delta.x, -delta.y, -delta.z));
        }
    }
}

// ===========================================================================
// cloth_updateClothContactValidityLaunch
// ===========================================================================
[shader("compute")]
[numthreads(CLOTH_STEP, 1, 1)]
void cloth_updateClothContactValidityLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint id = Ptr<uint>(gUpdate.activeIds)[groupId.y];
    Ptr<PxgFEMClothGpu> femCloth = clothAt(gUpdate.femCloths, id);
    const uint nbVerts = femCloth[0].mNbVerts;

    const uint64_t curPositions = femCloth[0].mPosition_InvMass;
    const uint64_t prevPositionsInContactOffset = femCloth[0].mPrevPositionInContactOffset;

    const uint globalThreadIndex = groupThreadId.x + CLOTH_STEP * groupId.x;

    const bool adaptive = (gUpdate.adaptiveFlagWord & 0xFFu) != 0u;
    const bool updateAdaptiveCollisionPair = adaptive && ((Ptr<uint>(gUpdate.updateContactPairs)[0] & 0xFFu) == 1u);

    if (updateAdaptiveCollisionPair && globalThreadIndex == 0u && groupId.y == 0u)
    {
        // Reset the contact count.
        Ptr<uint>(gUpdate.VTContactCount)[0] = 0u;
        Ptr<uint>(gUpdate.EEContactCount)[0] = 0u;
    }

    if (globalThreadIndex < nbVerts)
    {
        float4 curPos = Ptr<float4>(curPositions)[globalThreadIndex];
        if (updateAdaptiveCollisionPair)
        {
            curPos.w = 1.0f;   // Mark prevPosInContactOffset as set.
            Ptr<float4>(prevPositionsInContactOffset)[globalThreadIndex] = curPos;
        }
    }
}
