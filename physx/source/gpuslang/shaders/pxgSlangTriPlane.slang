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

// pxgSlangTriPlane.slang -- P9.82.
//
// The DECISION CORE of trimeshPlaneNarrowphase (trimeshCollision.cu:63-430),
// lifted out of npTriPlaneNp.slang so that the Phase-1 fixture and the engine
// kernel (triPlaneNpBda.slang) compile THE SAME CODE rather than two
// transcriptions of it.
//
// WHY THIS MODULE EXISTS AND WHERE ITS EDGE IS.  P9.15d's lesson was that a
// differential comparing two shaders which each carry their own copy of a body
// measures the copies' drift, not the glue it is aimed at.  The same hazard
// applies one step earlier: a Phase-1 fixture that certifies its own
// transcription certifies nothing about the kernel the engine actually runs.
//
// So everything that DECIDES something is here -- the reduction, the clustering,
// the batching loop's faithful overshoot, the margin, the manifold predicates --
// and everything that merely says WHERE THE BYTES ARE stays in the callers.  That
// split is deliberate: the fixture reads StructuredBuffers and the engine kernel
// reads Ptr<> through buffer device addresses, and forcing one of them onto the
// other's access model would have meant rewriting a 935-line verified fixture to
// share code that does no arithmetic.  Every float operation a port could get
// subtly wrong is in this file; no buffer access is.
//
// The bodies below are VERBATIM from npTriPlaneNp.slang, which P9.82 re-pointed
// at this module in the same commit -- they were moved, not retyped.

module pxgSlangTriPlane;

import pxgSlangCommon;
import pxgSlangPcmManifold;

public static const uint PLANE_TRI_MAX_CONTACTS = 6u;
public static const float MESH_MANIFOLD_EPSILON = 0.05f;
public static const float BOX_MARGIN_RATIO = 0.25f;
public static const float FLT_MAX_LOCAL = 3.402823466e38f;

public float3 vertex2Shape(float3 v, float3 scale, float4 rot)
{
    return quatRotateInv(rot, quatRotate(rot, v) * scale);
}
public float calculatePCMConvexMargin(float3 extents_, float3 scale, float toleranceLength)
{
    const float3 extents = extents_ * scale;
    const float mn = min(min(extents.x, extents.y), extents.z);
    return min(mn * BOX_MARGIN_RATIO, toleranceLength * MESH_MANIFOLD_EPSILON);
}

// NOTE: with an empty mask (or all masked values at the sentinel) these
// return idx 0, unlike CUDA maxIndex/minIndex (lowest MASKED index); every
// call site below provably passes a nonempty mask with finite values
public uint lowestArgMax(float vals[32], uint mask, out float best)
{
    best = -FLT_MAX_LOCAL;
    uint idx = 0;
    for (uint j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] > best) { best = vals[j]; idx = j; }
    return idx;
}
public uint lowestArgMin(float vals[32], uint mask, out float best)
{
    best = FLT_MAX_LOCAL;
    uint idx = 0;
    for (uint j = 0; j < 32; ++j)
        if (((mask >> j) & 1) != 0 && vals[j] < best) { best = vals[j]; idx = j; }
    return idx;
}

// contactReduce<clustering=true, keepDeepestPCM=true, maxPoints=6, dupeTest=false>
// (contactReduction.cuh via the tranche-82 transcription), on lane arrays
public uint contactReduce6(float3 ptRaw[32], float sep[32], float3 normal, uint allMask, float clusterBias)
{
    const int maxPoints = int(PLANE_TRI_MAX_CONTACTS);
    float3 point[32];
    for (uint j = 0; j < 32; ++j)
        point[j] = ptRaw[j] - normal * dot(ptRaw[j], normal);

    const uint newCount = countbits(allMask);
    if (newCount <= uint(maxPoints))
        return allMask;

    float v;
    float vals[32];

    for (uint j = 0; j < 32; ++j) vals[j] = sep[j];
    const uint i0 = lowestArgMax(vals, allMask, v);
    uint mask = 1u << i0;

    for (uint j = 0; j < 32; ++j)
    {
        const float3 d = point[i0] - point[j];
        vals[j] = sqrt(dot(d, d)); // PxVec3::magnitude(); sqrt(dot) keeps the
                                   // IEEE-sqrt assumption minimal vs OpLength
    }
    const uint i1 = lowestArgMax(vals, allMask & ~mask, v);
    mask |= 1u << i1;

    const float3 dir = cross(normal, point[i1] - point[i0]);
    float dval[32];
    for (uint j = 0; j < 32; ++j) dval[j] = dot(dir, point[j] - point[i0]);

    for (uint j = 0; j < 32; ++j) vals[j] = dval[j];
    const uint f = lowestArgMax(vals, allMask & ~mask, v);
    mask |= 1u << f;

    for (uint j = 0; j < 32; ++j) vals[j] = dval[j];
    const uint g = lowestArgMin(vals, allMask & ~mask, v);
    mask |= 1u << g;

    if (countbits(mask) == 4 && maxPoints > 4)
    {
        const uint i4 = lowestArgMin(sep, allMask & ~mask, v);
        mask |= 1u << i4;
    }

    // clustering
    {
        float sepB[32];
        for (uint j = 0; j < 32; ++j)
            sepB[j] = sep[j] - (((mask >> j) & 1) != 0 ? clusterBias : 0.0f);

        int label[32];
        int nbClusters = 0;
        {
            uint m = mask;
            float tbest[32];
            for (uint j = 0; j < 32; ++j) { tbest[j] = FLT_MAX_LOCAL; label[j] = -1; }
            for (; m != 0; ++nbClusters, m &= (m - 1))
            {
                const uint anchor = firstbitlow(m);
                for (uint j = 0; j < 32; ++j)
                {
                    const float3 dv = point[j] - point[anchor];
                    const float d = dot(dv, dv);
                    if (d < tbest[j]) { tbest[j] = d; label[j] = nbClusters; }
                }
            }
        }

        mask = 0;
        for (int i = 0; i < nbClusters; ++i)
        {
            uint cluster = 0;
            for (uint j = 0; j < 32; ++j)
                if (label[j] == i) cluster |= 1u << j;
            cluster &= allMask;
            if (cluster != 0)
                mask |= 1u << lowestArgMin(sepB, cluster, v);
        }
        for (int i = nbClusters; i < maxPoints; ++i)
            mask |= 1u << lowestArgMin(sepB, allMask & ~mask, v);
    }
    return mask;
}


// ---------------------------------------------------------------------------
// THE BATCHING LOOP (trimeshCollision.cu:335-395), the piece most likely to be
// retyped wrong, and the only one with a deliberate quirk.
//
// Warp 0 re-reduces the whole shared buffer in rounds.  Lanes that survived the
// previous round KEEP their point; free lanes refill from the buffer at
// `readIndex = exclusive-scan of the free-lane mask + ind`.  Then
//
//     ind += popc(readMask)
//
// advances by the number of FREE LANES, whether or not each of them actually read
// -- so near the end of the buffer `ind` can OVERSHOOT totalContacts.  That is
// upstream's arithmetic and it terminates the loop one round earlier than a
// "count what was actually read" version would.  Transcribed, not tidied.
//
// Returns the number of survivors, compacted to the FRONT of contacts/seps.
public uint tpBatchReduce(inout float3 contacts[198], inout float seps[198],
                          uint totalContacts, float3 worldNormal, float clusterTolerance)
{
    float3 lanePt[32];
    float laneSep[32];
    bool laneHas[32];
    for (uint l = 0; l < 32; ++l) { laneHas[l] = false; lanePt[l] = float3(0, 0, 0); laneSep[l] = FLT_MAX_LOCAL; }
    for (uint ind = 0; ind < totalContacts;)
    {
        uint readMask = 0;
        for (uint l = 0; l < 32; ++l)
            if (!laneHas[l]) readMask |= 1u << l;
        for (uint l = 0; l < 32; ++l)
        {
            if (laneHas[l]) continue;
            const uint readIndex = countbits(readMask & ((1u << l) - 1u)) + ind;
            if (readIndex < totalContacts)
            {
                lanePt[l] = contacts[readIndex];
                laneSep[l] = seps[readIndex];
                laneHas[l] = true;
            }
        }
        uint mask = 0;
        for (uint l = 0; l < 32; ++l)
            if (laneHas[l]) mask |= 1u << l;
        ind += countbits(readMask); // faithful: advances by FREE lanes, read or not
        mask = contactReduce6(lanePt, laneSep, worldNormal, mask, clusterTolerance);
        for (uint l = 0; l < 32; ++l)
            laneHas[l] = ((mask >> l) & 1) != 0;
    }
    uint k2 = 0;
    for (uint l = 0; l < 32; ++l)
    {
        if (!laneHas[l]) continue;
        contacts[k2] = lanePt[l];
        seps[k2] = laneSep[l];
        ++k2;
    }
    return k2;
}

// The two manifold predicates live in pxgSlangPcmManifold, which already carried
// them for the BDA callers.  P9.82's first draft gave them a SECOND home here
// before noticing that -- the very duplication this module exists to prevent, made
// by the commit that made the module.  These two are thin adapters onto that one
// spelling, for the caller that holds the previous transform in registers.
public bool tpInvalidateManifold(float4 relQ, float3 relP, float4 prevQ, float3 prevP,
                                 float thresholdP)
{
    return mmInvalidatePredicate(relQ, relP, prevQ, prevP, thresholdP);
}

public bool tpRefreshContact(float4 relQ, float3 relP, float3 pointA, float3 pointB,
                             inout float4 nPen, float sqThreshold)
{
    float penNew;
    bool lost;
    mmRefreshMath(relQ, relP, pointA, pointB, nPen.xyz, nPen.w, sqThreshold, penNew, lost);
    nPen.w = penNew;
    return lost;
}
