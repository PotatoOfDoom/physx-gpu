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

// fcVertexConvex.slang
//
// The Phase-1 battery for the cloth VERTEX vs convex contact path
// (femClothPrimitives.cu clothVertexConvexCollision + closestVertToConvex +
// computeClosestPolygon + distancePointSegment, plus the 5.12
// deformableElementFilter.cuh findRigidSurfaceVertexFilter).
//
// P9.21 -- THE ALGEBRA MOVED TO pxgSlangVertexConvexCloth.  This shader keeps
// its entry point and its battery, which is what verifies the lift: the battery
// compares against a C++ host mirror of the CUDA, so it goes on measuring the
// core against an INDEPENDENT reference after the move, not against a second
// copy of itself.  What remains here is exactly what the engine kernel
// cloth_convexVertexContactGenLaunch also has to supply for itself:
//
//   * the LIVE attachment-filter check (PhysX 5.12) -- findRigidSurfaceVertexFilter:
//     ONE exact binary search of the sorted PxgDeformablePairInfo list for
//     (PxgEncodeRigidFilterRef(rigidId, rigidCacheRef), PxgEncodeDeformableVertex(
//     clothId, vertexIdx)), compared ref0 then ref1 as unsigned 64-bit.  A filtered
//     vertex generates NOTHING.  The rigid cache ref rides in FvcCase.ids2.z;
//   * the case: the world cloth vertex, the rigid pose, the convex scale and
//     scale rotation, the ids and the two distances.
//
// The hull is read through pxgSlangConvexHull's BDA accessors inside the
// module.  Its section offsets were checked against this file's former
// uint-offset accessors, which are now gone, byte for byte:
//     hullDesc        = hull + 16                              (both)
//     vertices        = hull + 48                              (both)
//     planes          = vertices + 16*nbVerts                  (both)
//     polyDescs       = planes   + 16*nbPolys                  (both)
//     verticesByEdges = polyDescs + 4*nbPolys                  (both)
//     facesByEdges    = verticesByEdges + 4*nbEdges            (both)
//     vertexData8     = facesByEdges + 2*nbEdges + 3*nbVerts   (both)
// and the field decodes agree too (nbPolygons = desc & 0xff, nbVerts =
// (desc >> 8) & 0xff, nbEdges = desc >> 16; polyDesc nbVerts = (d >> 8) & 0xff,
// vRef8 = d >> 16).

import pxgSlangCommon;
// The lifted core.  FvcCase lives there -- one definition, two users: this battery
// and the engine-ABI kernel.
import pxgSlangVertexConvexCloth;
import pxgSlangDeformableRef;       // PxgEncodeRigidFilterRef, PxgComparePairInfo
import pxgSlangDeformableWriters;   // PxgRigidDbContactWriterGpu

struct FvcParams
{
    uint numCases; uint maxContacts; uint nbFilterPairs; uint pad0;   // @0
    uint64_t hullAddr;        // @16  base of the hull blob (FvcCase.ids2.y is a UINT offset into it)
    // PxgRigidDbContactWriter, by address (@24..@96)
    uint64_t wPoint, wNormalPen, wBary, wInfo, wCount, wByX, wTempByX, wIdxByX;
    uint64_t wRigidMat;       // @88  PxU16* outRigidMaterialIndex (5.12)
}

// PxgDeformablePairInfo (16 bytes): u64 ref0 (x lo, y hi), u64 ref1 (z lo, w hi).
typealias FilterPair = uint4;

[[vk::push_constant]]
ConstantBuffer<FvcParams> gParams;

RWStructuredBuffer<FvcCase> gCase;        // binding 0 (test input; not the engine ABI)
RWStructuredBuffer<FilterPair> gFilter;   // binding 1 (sorted)
RWStructuredBuffer<float4> gDebug;        // binding 2: (filtered, pen, wrote, insideBranch)

// ---- find(const PxgDeformablePairInfo*, nbPairs, pair) (deformableElementFilter.cuh, 5.12)
bool findFilter(uint64_t ref0, uint64_t ref1)
{
    const PxgDeformablePairInfoGpu pair = PxgMakePairInfo(ref0, ref1);
    uint l = 0, r = gParams.nbFilterPairs;
    while (l < r)
    {
        const uint mid = (l + r) / 2;
        const uint4 e = gFilter[mid];
        const PxgDeformablePairInfoGpu p = PxgMakePairInfo(uint64_t(e.x) | (uint64_t(e.y) << uint64_t(32)),
                                                           uint64_t(e.z) | (uint64_t(e.w) << uint64_t(32)));
        const int cmp = PxgComparePairInfo(pair, p);
        if (cmp == 0)
            return true;
        if (cmp < 0)
            r = mid;
        else
            l = mid + 1;
    }
    return false;
}

// ---------------------------------------------------------------------------
[shader("compute")]
[numthreads(64, 1, 1)]
void ctFcVertexConvex(uint3 tid : SV_DispatchThreadID)
{
    const uint caseIdx = tid.x;
    if (caseIdx >= gParams.numCases) return;
    const FvcCase cs = gCase[caseIdx];

    // LIVE attachment filter (a hit generates nothing)
    // findRigidSurfaceVertexFilter(cloth filters, rigidId, rigidCacheRef, clothId, vertexIdx)
    const uint64_t rigidId = uint64_t(cs.ids.x) | (uint64_t(cs.ids.y) << uint64_t(32));
    if (findFilter(PxgEncodeRigidFilterRef(rigidId, cs.ids2.z), PxgEncodeDeformableVertex(cs.ids.z, cs.ids.w)))
    {
        gDebug[caseIdx] = float4(1, 0, 0, 0);
        return;
    }

    // FvcCase.ids2.y stays a UINT offset into the blob; the address is formed here.
    const uint64_t hull = gParams.hullAddr + uint64_t(cs.ids2.y) * 4UL;

    PxgRigidDbContactWriterGpu w;
    w.outPoint = gParams.wPoint; w.outNormalPen = gParams.wNormalPen; w.outBarycentric = gParams.wBary;
    w.outContactInfo = gParams.wInfo; w.outRigidMaterialIndex = gParams.wRigidMat;
    w.totalContactCount = gParams.wCount;
    w.contactByRigid = gParams.wByX; w.tempContactByRigid = gParams.wTempByX;
    w.contactIndexSortedByRigid = gParams.wIdxByX;
    w.maxNumContacts = gParams.maxContacts;

    gDebug[caseIdx] = fcClothVertexConvexContactGen(cs, hull, w);
}
