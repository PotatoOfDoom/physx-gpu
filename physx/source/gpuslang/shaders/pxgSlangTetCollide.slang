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

// pxgSlangTetCollide — tetPrimitivesCollide2 (convexNpCommon.h:212), the GJK-then-EPA
// driver every deformable-vs-convex narrowphase path in PhysX goes through.
//
// It is thirty lines of policy on top of the two modules that do the work:
//
//   1. seed GJK from the caller's search direction, falling back to +x when it is zero;
//   2. run GJK.  DISTANT ends it -- the pair is further apart than the contact distance;
//   3. decide whether the GJK answer is trustworthy.  It is not when the shapes OVERLAP
//      (GJK cannot measure penetration), and it is not when they are merely CLOSE but the
//      witness direction and the search direction disagree by more than 0.999 -- upstream
//      calls that an ANOMALY, and it is what catches a nearly-degenerate simplex that
//      happened to converge;
//   4. in either case run EPA from GJK's final simplex;
//   5. if EPA itself reports degenerate, throw the cache away and run the whole pair AGAIN
//      from a different seed (+y).  Upstream's comment is "we need to re-run gjk epa with
//      other configurations"; the retry is not a refinement, it is a different starting
//      simplex for the same query.
//
// THE OUTPUT IS SHARED, WHICH IS WHY EpaOut IS FOLDED BACK INTO GjkOut.  CUDA has ONE
// `GjkOutput` struct that gjk() fills and epa() then overwrites in place
// (ss_scratch.gjkOutput).  The two ports gave it two shapes, so the fields EPA actually
// writes are copied across explicitly here; `size` and `closestPointDir` are GJK's alone and
// keep their values, exactly as in the .cu.
//
// The convergence ratio is upstream's constant, and EPA's epsilon is derived from the two
// inner-sphere radii the caller supplies -- for a tetrahedron that is computeInnerSphere's
// result, for a sphere or capsule it is zero, and for a hull it is the stored extent scaled
// by the smallest axis scale.
module pxgSlangTetCollide;

import pxgSlangGjk;
import pxgSlangEpa;

// convexNpCommon.h:219.
public static const float TETC_CONVERGENCE_RATIO = 1.0f - 0.000225f;

// EPA writes a subset of the shared GjkOutput; copy exactly that subset.
void foldEpaIntoGjk(EpaOut e, inout GjkOut o)
{
    o.degenerate = e.degenerate;
    o.lowerBound = e.lowerBound;
    o.upperBound = e.upperBound;
    o.direction = e.direction;
    o.closestPointA = e.closestPointA;
    o.closestPointB = e.closestPointB;
}

public uint tetPrimitivesCollide2(GjkCloud a, GjkCloud b, float3 searchDir, float contactDistance,
                                  float inSphereRadius0, float inSphereRadius1, out GjkOut o)
{
    float3 initialDir = float3(1.0f, 0.0f, 0.0f);
    // GJK lazily normalises its start direction, so an unnormalised search vector is fine --
    // upstream passes it straight through and only guards the zero case.
    if (dot(searchDir, searchDir) > 0.0f)
        initialDir = searchDir;

    const float relEpsBase = 0.5f * (inSphereRadius0 + inSphereRadius1);
    const float releps = relEpsBase * (1.0f - TETC_CONVERGENCE_RATIO);

    float cached[32];
    const uint gjkResult = gjkRun(a, b, initialDir, contactDistance, TETC_CONVERGENCE_RATIO, o, cached);

    if (gjkResult == GJK_DISTANT)
        return GJK_DISTANT;

    const bool separated = (gjkResult == GJK_CLOSE);
    const bool anomaly = separated && dot(o.closestPointDir.xyz, o.direction.xyz) < 0.999f;

    uint result = GJK_CLOSE;

    if (!separated || anomaly)
    {
        EpaOut e;
        result = epaRun(cached, o.size, releps, a, b, e);
        foldEpaIntoGjk(e, o);

        if (o.degenerate != 0u)
        {
            // Throw the cached simplex away and re-seed from +y.  gjkRun is always a cold
            // start, so "cachedData.size = 0" needs no expression here.
            float cached2[32];
            gjkRun(a, b, float3(0.0f, 1.0f, 0.0f), contactDistance, TETC_CONVERGENCE_RATIO, o, cached2);

            EpaOut e2;
            result = epaRun(cached2, o.size, releps, a, b, e2);
            foldEpaIntoGjk(e2, o);
        }
    }

    return result;
}

// computeInnerSphere (softbodyPrimitives.cu:123).  NOT the inscribed sphere in the textbook
// sense: upstream takes the CENTROID and the smallest absolute distance from it to the four
// face planes.  Transcribed rather than re-derived -- an incentre formula gives a different
// number for any non-regular tetrahedron, and this value feeds EPA's convergence epsilon.
//
// CUDA computes it cooperatively: 12 lanes hold the four vertices' components and
// __shfl_sync reassembles them (softbodyPrimitives.cu:235).  With one thread per pair the
// vertices are simply in hand, so the shuffle stage disappears; the arithmetic below is the
// 4-vector overload (:123) it delegates to, face winding for face winding.
float3 tcNormalized(float3 v)
{
    const float m = dot(v, v);
    if (m > 0.0f)
    {
        const float rs = 1.0f / sqrt(m);
        return v * rs;
    }
    return float3(0.0f, 0.0f, 0.0f);
}

public float computeInnerSphere(float3 v0, float3 v1, float3 v2, float3 v3, out float3 tetCenter)
{
    const float3 center = (v0 + v1 + v2 + v3) * 0.25f;
    tetCenter = center;

    // 0 2 1
    float3 n = tcNormalized(cross(v2 - v0, v1 - v0));
    float innerSphere = abs(dot(center - v0, n));

    // 1 2 3
    n = tcNormalized(cross(v2 - v1, v3 - v1));
    innerSphere = min(abs(dot(center - v1, n)), innerSphere);

    // 0 1 3
    n = tcNormalized(cross(v1 - v0, v3 - v0));
    innerSphere = min(abs(dot(center - v0, n)), innerSphere);

    // 0 3 2
    n = tcNormalized(cross(v3 - v0, v2 - v0));
    innerSphere = min(abs(dot(center - v0, n)), innerSphere);

    return innerSphere;
}
