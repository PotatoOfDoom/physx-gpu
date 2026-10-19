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
// pxgSlangClothTriContact.slang -- one STATIC TRIANGLE against a cloth triangle,
// shared by the cloth contact generators (P9.27b).
//
// WHY THIS IS A MODULE AND NOT A THIRD COPY
//
// staticTriangleVsClothCollision (femClothPrimitives.cu:3605) is ONE __device__
// function upstream, called by cloth_meshContactGenLaunch (:3749) and
// cloth_heightfieldContactGenLaunch (:3643).  The only thing that differs between
// the two callers is where the static triangle comes from -- three vertices out of
// a cooked mesh's arrays, or three out of a heightfield's samples -- and both hand
// it to this in the CLOTH's space.
//
// So no similarity had to be measured here, unlike pxgSlangHeightfield: there is
// one function, and the second port simply had not been written yet.  P9.27b is
// the moment it would have been, which is the moment to lift instead.
//
// The body arrived in this port as a LIFT of fcStaticTriCloth.slang's mode 0 --
// already fixture-verified against a CPU transcription for the descriptor-set
// engine -- and P8.17 copied it into femClothMeshContactGenBda.  That copy is what
// moves here.  Phase 1 kept its own copy until the 5.12 port, when fcStaticTriCloth
// became a harness around THIS module: its random battery (testFcStaticTriCloth.cpp)
// now lands on the code the engine ships, not on a copy.
//
// WHAT IS IN HERE is the clipping pipeline (trimLoop, trimTriangleByTriangle,
// getDeepestPointSimplified), the small plane/barycentric helpers it needs, and
// the FemContact it produces.  What is NOT is the pair-stack walk, the shape
// routing and the writer -- those are the caller's glue and they differ.
// ===========================================================================

module pxgSlangClothTriContact;

import pxgSlangCommon;   // quatRotate / quatRotateInv

// PxTolerancesScale-independent normalization floor (PxVec3::getNormalized).
public static const float PX_NORMALIZATION_EPSILON = 1e-20f;
// ---------------------------------------------------------------------------
// Small geometry helpers (fcStaticTriCloth.slang, verbatim).
// ---------------------------------------------------------------------------
public float3 cgGetNormalized(float3 v)
{
    const float m = dot(v, v);
    return m > 0.0f ? v * rsqrt(m) : float3(0, 0, 0);
}
// PxPlane(point, normal): n, d = -p.n ; distance(p) = n.p + d
public float4 makePlane(float3 p, float3 n) { return float4(n, -dot(p, n)); }
public float planeDistance(float4 pl, float3 p) { return dot(pl.xyz, p) + pl.w; }

// vertex2Shape(v, scale, rot) = rot.rotateInv(rot.rotate(v) * scale)  (cudaNpCommon.h:270).
// MULTIPLIES by the scale — the inverse of step 1's shape2Vertex, which divides.
public float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}
public float3 transformPt(float4 q, float3 p, float3 v) { return quatRotate(q, v) + p; }

// femClothPrimitives.cu computeBarycentric (the bool result is unused at this call site).
public void computeBarycentric(float3 a, float3 b, float3 c, float3 p, out float4 barycentric)
{
    const float3 v0 = b - a;
    const float3 v1 = c - a;
    const float3 v2 = p - a;
    const float d00 = dot(v0, v0);
    const float d01 = dot(v0, v1);
    const float d11 = dot(v1, v1);
    const float d20 = dot(v2, v0);
    const float d21 = dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    const float v = (d11 * d20 - d01 * d21) / denom;
    const float w = (d00 * d21 - d01 * d20) / denom;
    const float u = 1.0f - v - w;
    barycentric = float4(u, v, w, 0.0f);
}

// ---------------------------------------------------------------------------
// The polygon trim pipeline (femClothPrimitives.cu:3436-3641), lifted from
// fcStaticTriCloth.slang where it is already fixture-verified against a CPU transcription.
// ---------------------------------------------------------------------------

// Only correct when a and b straddle the plane (one may lie exactly on it).
public float3 intersectionPoint(float4 plane, float3 a, float3 b)
{
    const float distA = abs(planeDistance(plane, a));
    const float distB = abs(planeDistance(plane, b));
    const float f = distA / (distA + distB);
    return (1.0f - f) * a + f * b;
}

public void insertPt(inout float3 arr[8], int arrCount, int index, float3 element)
{
    for (int i = arrCount; i > index; --i)
        arr[i] = arr[i - 1];
    arr[index] = element;
}

// In-place convex-loop clip: points with distance > 0 (STRICT) are removed.
public int trimLoop(float4 plane, inout float3 loop[8], int loopCount)
{
    if (loopCount < 3)
        return loopCount;

    float3 intersectionA = float3(0, 0, 0);
    int changeA = -1;
    float3 intersectionB = float3(0, 0, 0);
    int changeB = -1;

    bool keep = false;

    bool prevOutside = planeDistance(plane, loop[0]) > 0.0f;
    for (int i = 0; i < loopCount; ++i)
    {
        const bool outside = planeDistance(plane, loop[(i + 1) % loopCount]) > 0.0f;
        if (outside != prevOutside)
        {
            const float3 inter = intersectionPoint(plane, loop[i], loop[(i + 1) % loopCount]);
            if (changeA < 0)
            {
                changeA = i;
                keep = !prevOutside;
                intersectionA = inter;
            }
            else
            {
                changeB = i;
                intersectionB = inter;
            }
        }
        prevOutside = outside;
    }

    if (changeA >= 0 && changeB >= 0)
    {
        int loopIndexer = -1;
        for (int i = 0; i < loopCount; ++i)
        {
            if (keep)
            {
                ++loopIndexer;
                loop[loopIndexer] = loop[i];
            }
            if (i == changeA || i == changeB)
            {
                const float3 pt = (i == changeA) ? intersectionA : intersectionB;
                if (loopIndexer == i && !keep)
                {
                    ++loopIndexer;
                    insertPt(loop, loopCount, loopIndexer, pt);
                    ++loopCount;
                    ++i;
                    ++changeB;
                }
                else
                {
                    ++loopIndexer;
                    loop[loopIndexer] = pt;
                }
                keep = !keep;
            }
        }
        loopCount = loopIndexer + 1;
    }
    else if (prevOutside)
        loopCount = 0;   // no crossing and every point outside

    return loopCount;
}

public float4 createEdgePlane(float3 triNormal, float3 edgeStart, float3 edgeEnd)
{
    const float3 n = cgGetNormalized(cross(edgeEnd - edgeStart, triNormal));
    return makePlane(edgeStart, n);
}

// The 5-plane, 8-element variant: three edge planes plus the two +-distance slabs.
public int trimTriangleByTriangle(inout float3 poly[8], int polyLength, float3 triA, float3 triB, float3 triC,
                           float maxDistanceLimitPosNormalDir, float maxDistanceLimitNegNormalDir)
{
    float3 n = cross(triB - triA, triC - triA);
    // PxVec3::normalizeSafe: n is left UNCHANGED below the epsilon, unlike getNormalized which
    // returns zero.  Only distinguishable for 0 < |cross| < 1e-20, a subnormal window that
    // cannot survive the vertex transform chain — kept because it is what the .cu writes.
    const float mag = sqrt(dot(n, n));
    if (mag >= PX_NORMALIZATION_EPSILON)
        n = n * (1.0f / mag);

    polyLength = trimLoop(createEdgePlane(n, triA, triB), poly, polyLength);
    polyLength = trimLoop(createEdgePlane(n, triB, triC), poly, polyLength);
    polyLength = trimLoop(createEdgePlane(n, triC, triA), poly, polyLength);

    polyLength = trimLoop(makePlane(triA + maxDistanceLimitPosNormalDir * n, n), poly, polyLength);
    polyLength = trimLoop(makePlane(triA - maxDistanceLimitNegNormalDir * n, -n), poly, polyLength);

    return polyLength;
}

// Running min with a +-tolerance averaging band: a strictly deeper point RESTARTS the average,
// a near-tie ACCUMULATES into it without moving minDist.  A fully clipped-away polygon leaves
// minDist at 1e9, which fails the `sep < cDistance` gate (the 0/0 in the average is unused).
public float3 getDeepestPointSimplified(float3 poly[8], int polyLength, float4 plane, float tolerance,
                                 out float minDist)
{
    minDist = 1000000000.0f;
    float3 deepestPoint = float3(0, 0, 0);
    int count = 0;
    for (int i = 0; i < polyLength; ++i)
    {
        const float d = planeDistance(plane, poly[i]);
        if (d < minDist - tolerance)
        {
            minDist = d;
            deepestPoint = poly[i];
            count = 1;
        }
        else if (d < minDist + tolerance)
        {
            deepestPoint += poly[i];
            count++;
        }
    }
    // P9.36: femClothPrimitives.cu:3591 is `deepestPoint /= count;` on a PxVec3, i.e.
    // PxVec3::operator/= (PxVec3.h:260) -- ONE reciprocal, three multiplies.
    deepestPoint = deepestPoint * (1.0f / float(count));
    return deepestPoint;
}

public struct FemContact
{
    public float3 position;
    public float3 normal;
    public float separation;
    public float4 barycentric;
}

// staticTriangleVsClothCollision (femClothPrimitives.cu:3605).
public bool staticTriangleVsClothCollision(float3 localA0, float3 localB0, float3 localC0,
                                    float3 a1, float3 b1, float3 c1,
                                    float4 clothQ, float3 clothP, float cDistance,
                                    out FemContact contact)
{
    float3 convexPoly[8];
    convexPoly[0] = a1;
    convexPoly[1] = b1;
    convexPoly[2] = c1;
    for (int k = 3; k < 8; ++k)
        convexPoly[k] = float3(0, 0, 0);

    const float3 fixNormal = cgGetNormalized(cross(localB0 - localA0, localC0 - localA0));
    const float4 staticTrianglePlane = makePlane(localA0, fixNormal);

    const int numPolyCorners =
        trimTriangleByTriangle(convexPoly, 3, localA0, localB0, localC0, cDistance, cDistance);

    float sep;
    const float3 deepestPoint = getDeepestPointSimplified(convexPoly, numPolyCorners,
                                                          staticTrianglePlane, 0.01f * cDistance, sep);

    const bool hasContact = sep < cDistance;

    contact.position = float3(0, 0, 0);
    contact.normal = float3(0, 0, 0);
    contact.separation = 0.0f;
    contact.barycentric = float4(0, 0, 0, 0);
    if (hasContact)
    {
        computeBarycentric(a1, b1, c1, deepestPoint, contact.barycentric);
        contact.position = transformPt(clothQ, clothP, deepestPoint);
        contact.normal = quatRotate(clothQ, fixNormal);
        contact.separation = sep;
    }
    return hasContact;
}

// PxgFEMContactWriter::writeRigidVsDeformableContact32 (PxgFEMCore.h:501).  The 32-bit sort
// lane only: `contactSortedByX` takes the LOW WORD of the rigid node index and the u64
// `contactByX` / `tempContactByX` keys are deliberately left alone — this path's radix sort
// does not read them.  setRigidMaterialIndex(mat) followed by markInCollision(false) collapses
// to a single store of (mat & 0x7fffffff).
