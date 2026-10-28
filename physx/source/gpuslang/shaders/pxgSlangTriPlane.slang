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
import pxgSlangContactReduceSerial;   // P9.83b-2: the reduction body, shared

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

// P9.83b-2: these moved into pxgSlangContactReduceSerial with the reduction body
// that uses them, and are re-exported here because npTriPlaneNp and triPlaneNpBda
// name them.  Thin forwarders, not a second implementation.
public uint lowestArgMax(float vals[32], uint mask, out float best)
{
    return crsArgMax(vals, mask, best);
}
public uint lowestArgMin(float vals[32], uint mask, out float best)
{
    return crsArgMin(vals, mask, best);
}

// contactReduce<clustering=true, keepDeepestPCM=true, maxPoints=6, dupeTest=false>
// (contactReduction.cuh, wrapper 1), on lane arrays.
//
// P9.83b-2: THE BODY MOVED OUT.  It is contactReduceShared, which is ONE CUDA
// template that this corpus had transcribed four times -- once per instantiation
// -- and a fifth was about to be written for triangleTriangleCollision's
// contactReduce2 wrapper.  pxgSlangContactReduceSerial now holds it once, with
// the template arguments as a CrsConfig and the wrapper's two inputs as
// parameters.  This is a thin adapter; its two fixtures re-ran bit-exactly across
// the move, which is the evidence the transcription was faithful, not a reason to
// have kept it here.
public uint contactReduce6(float3 ptRaw[32], float sep[32], float3 normal, uint allMask, float clusterBias)
{
    return crsContactReduce(ptRaw, sep, normal, allMask, clusterBias,
                            crsConfig(true, true, int(PLANE_TRI_MAX_CONTACTS), false));
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
// Returns the number of survivors, compacted to the FRONT of the store.
//
// THE LOOP IS GENERIC OVER WHERE THE BUFFER LIVES.  The Phase-1 fixture keeps the
// CUDA's shared `contacts`/`separations` buffer in per-thread arrays; the engine
// kernel (triPlaneNpBda) keeps it in groupshared, because since its stage 1 runs on
// the whole 1024-thread group the buffer is FILLED by many threads and only this
// loop runs on one.  Both compile this one body -- the storage is the only thing an
// ITpContactStore supplies, so the readMask/overshoot arithmetic cannot drift
// between the two (the P9.15d rule this module exists for).
public interface ITpContactStore
{
    float3 tpPoint(uint i);
    float tpSep(uint i);
    [mutating] void tpStore(uint i, float3 p, float s);
}

public uint tpBatchReduceStore<S : ITpContactStore>(inout S store, uint totalContacts,
                                                    float3 worldNormal, float clusterTolerance)
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
                lanePt[l] = store.tpPoint(readIndex);
                laneSep[l] = store.tpSep(readIndex);
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
        store.tpStore(k2, lanePt[l], laneSep[l]);
        ++k2;
    }
    return k2;
}

// The per-thread array store (the Phase-1 fixture's shape).
struct TpArrayStore : ITpContactStore
{
    float3 pts[198];
    float seps[198];
    float3 tpPoint(uint i) { return pts[i]; }
    float tpSep(uint i) { return seps[i]; }
    [mutating] void tpStore(uint i, float3 p, float s) { pts[i] = p; seps[i] = s; }
}

public uint tpBatchReduce(inout float3 contacts[198], inout float seps[198],
                          uint totalContacts, float3 worldNormal, float clusterTolerance)
{
    TpArrayStore store;
    store.pts = contacts;
    store.seps = seps;
    const uint k2 = tpBatchReduceStore(store, totalContacts, worldNormal, clusterTolerance);
    contacts = store.pts;
    seps = store.seps;
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
