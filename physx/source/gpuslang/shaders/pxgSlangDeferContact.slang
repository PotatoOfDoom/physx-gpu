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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS \'\'AS IS\'\' AND ANY
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
//
// ===========================================================================
// pxgSlangDeferContact — delayGenerateContact (convexTriangle.cuh:1402).
//
// Step 2 of P8.31's corrected decomposition, and the counterpart of the
// classifier P8.30 certified.  P8.30 answers "WHICH feature did the contact land
// on"; this answers the prior question, "should this pair be deferred to the
// second pass AT ALL".
//
// The predicate, verbatim:
//
//     if (featureIndex != EDGE_FEATURE_IDX && featureIndex != TRI_FEATURE_IDX)
//     {
//         PxVec3 triLocNormal = -s_scratch->triangleLocNormal;
//         PxReal cosTheta = minNormal.dot(triLocNormal);
//         const PxReal threshold = 0.707106781f;   // about 45 degrees
//         return cosTheta < threshold;
//     }
//     return false;
//
// FOUR THINGS THAT ONLY LITERAL TRANSCRIPTION KEEPS:
//
//   1. THE GUARD IS AN AND OF TWO INEQUALITIES, and the two sentinels are NOT
//      adjacent values: TRI_FEATURE_IDX is 0x7FFFFFFF and EDGE_FEATURE_IDX is
//      0x80000000 (triangle.cuh:32/33).  They differ in every bit above the low
//      31, so no range check can stand in for the pair.  When the separating
//      axis IS the triangle face or an edge cross, the contact is never deferred
//      — only a CONVEX HULL FACE normal can defer.
//
//   2. THE TRIANGLE NORMAL IS NEGATED FIRST.  `cosTheta` is measured against
//      MINUS the stored triangle local normal.  Dropping the negation flips the
//      predicate on every pair whose axes are within 45 degrees of opposed,
//      which is most of them.
//
//   3. THE COMPARISON IS STRICTLY LESS.  A pair exactly at the 45-degree
//      threshold is NOT deferred.
//
//   4. THE THRESHOLD IS THE LITERAL 0.707106781f, not a computed
//      `sqrt(0.5)` and not the 0.985f that sits commented out one line below it
//      in upstream.  The commented value is a tighter, 10-degree bar someone
//      tried; reinstating it would defer far more pairs and change the contact
//      set on every mesh.
// ===========================================================================

module pxgSlangDeferContact;

// triangle.cuh:32/33
public static const uint TRI_FEATURE_IDX  = 0x7FFFFFFFu;
public static const uint EDGE_FEATURE_IDX = 0x80000000u;

// about 45 degrees; the literal from convexTriangle.cuh:1411
public static const float DEFER_COS_THRESHOLD = 0.707106781f;

// mirrors delayGenerateContact().  `triangleLocNormal` is the scratch field the
// CUDA reads; it is passed in rather than fetched so the predicate stays pure.
public bool dcDelayGenerateContact(float3 minNormal, float3 triangleLocNormal, uint featureIndex)
{
    if (featureIndex != EDGE_FEATURE_IDX && featureIndex != TRI_FEATURE_IDX)
    {
        const float3 triLocNormal = -triangleLocNormal;
        const float cosTheta = dot(minNormal, triLocNormal);
        return cosTheta < DEFER_COS_THRESHOLD;
    }
    return false;
}
