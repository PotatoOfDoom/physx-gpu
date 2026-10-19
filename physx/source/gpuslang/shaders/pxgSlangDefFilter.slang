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

// ===========================================================================
// pxgSlangDefFilter -- the deformable element filter, PhysX 5.12.
//
// Transcribed from gpunarrowphase/src/CUDA/deformableElementFilter.cuh (5.12).  The
// 5.6.1 design (one global sorted PxgRigidFilterPair / PxgNonRigidFilterPair list
// per launch, 32-bit packed ids, the ignoreIndex2 disjunct, the particle-buffer
// runsum chain) is gone.  Every deformable now carries one sorted
// PxgDeformablePairInfo array per pair type:
//
//   PxgSoftBody   mRigidVolumeFilterPairs   (rigid  , tet    )
//                 mVolumeVolumeFilterPairs  (vertex , tet    )
//                 mVolumeSurfaceFilterPairs (tet    , vertex | triangle)
//   PxgFEMCloth   mRigidSurfaceFilterPairs  (rigid  , vertex | triangle)
//                 mSurfaceSurfaceFilterPairs(vertex , triangle | edge, edge)
//
// and every lookup is ONE exact binary search for one fully encoded pair
// (PxgComparePairInfo: ref0 then ref1, unsigned 64-bit).  The ANY_FEATURE wildcard
// is not a search feature: callers that honour it issue a second (or third) exact
// search with PXG_DREF_ANY_FEATURE in the feature id, as the CUDA does.
//
// A rigid is keyed by PxgEncodeRigidFilterRef(node, transformCacheRef): bodies and
// links by their PxNodeIndex, statics per shape by the transform cache id.
//
// THE ZERO-LENGTH ARRAY.  A deformable with no filters of a kind has a NULL array
// and a zero count.  On llvmpipe a zero-trip `while (l < r)` still speculates the
// first iteration's load (P9.38, softBodySelfCollisionBda), and a load from address
// 0 faults.  So the search never sees the raw pointer: dfFilterArray() selects a
// caller-supplied safe address (the descriptor the count was read from) when the
// count is 0.  The search reads nothing from it -- the loop does not run -- and
// the answer is the CUDA's.
// ===========================================================================
module pxgSlangDefFilter;

import pxgSlangDeformableRef;
import pxgSlangFEMClothDesc;    // PxgFEMClothGpu: mRigidSurfaceFilterPairs, mSurfaceSurfaceFilterPairs
import pxgSlangSoftBodyDesc;    // PxgSoftBodyGpu: mRigidVolumeFilterPairs, mVolume*FilterPairs

public struct DfFilterArray
{
    public uint64_t pairs;  // PxgDeformablePairInfo*, never null
    public uint count;
}

// `safeAddr` must be a mapped address of at least 16 bytes (the owning descriptor).
public DfFilterArray dfFilterArray(uint64_t pairs, uint count, uint64_t safeAddr)
{
    DfFilterArray a;
    a.pairs = (count == 0u) ? safeAddr : pairs;
    a.count = count;
    return a;
}

// The per-actor arrays, read from the descriptor at `addr` (which doubles as the safe
// address).  The CUDA passes `femCloth.mRigidSurfaceFilterPairs,
// femCloth.mNumRigidSurfaceFilterPairs` at every call site; these are that pair.
public DfFilterArray clothRigidSurfaceFilters(uint64_t clothAddr)
{
    Ptr<PxgFEMClothGpu> c = Ptr<PxgFEMClothGpu>(clothAddr);
    return dfFilterArray(c[0].mRigidSurfaceFilterPairs, c[0].mNumRigidSurfaceFilterPairs, clothAddr);
}
public DfFilterArray clothSurfaceSurfaceFilters(uint64_t clothAddr)
{
    Ptr<PxgFEMClothGpu> c = Ptr<PxgFEMClothGpu>(clothAddr);
    return dfFilterArray(c[0].mSurfaceSurfaceFilterPairs, c[0].mNumSurfaceSurfaceFilterPairs, clothAddr);
}
public DfFilterArray softBodyRigidVolumeFilters(uint64_t softBodyAddr)
{
    Ptr<PxgSoftBodyGpu> b = Ptr<PxgSoftBodyGpu>(softBodyAddr);
    return dfFilterArray(b[0].mRigidVolumeFilterPairs, b[0].mNumRigidVolumeFilterPairs, softBodyAddr);
}
public DfFilterArray softBodyVolumeVolumeFilters(uint64_t softBodyAddr)
{
    Ptr<PxgSoftBodyGpu> b = Ptr<PxgSoftBodyGpu>(softBodyAddr);
    return dfFilterArray(b[0].mVolumeVolumeFilterPairs, b[0].mNumVolumeVolumeFilterPairs, softBodyAddr);
}
public DfFilterArray softBodyVolumeSurfaceFilters(uint64_t softBodyAddr)
{
    Ptr<PxgSoftBodyGpu> b = Ptr<PxgSoftBodyGpu>(softBodyAddr);
    return dfFilterArray(b[0].mVolumeSurfaceFilterPairs, b[0].mNumVolumeSurfaceFilterPairs, softBodyAddr);
}

// PxNodeIndex as the uint2 the port's remap-table reads produce.
public uint64_t dfNodeU64(uint2 node)
{
    return uint64_t(node.x) | (uint64_t(node.y) << uint64_t(32));
}

// find(const PxgDeformablePairInfo*, nbPairs, pair).
public bool dfFind(DfFilterArray arr, uint64_t ref0, uint64_t ref1)
{
    PxgDeformablePairInfoGpu pair = PxgMakePairInfo(ref0, ref1);
    uint l = 0u, r = arr.count;
    while (l < r)
    {
        const uint mid = (l + r) / 2u;
        const int cmp = PxgComparePairInfo(pair, PxgLoadPairInfo(arr.pairs, mid));

        if (cmp == 0)
            return true;
        if (cmp < 0)
            r = mid;
        else
            l = mid + 1u;
    }
    return false;
}

// Volume against volume: ref0 a vertex of the probing volume's collision mesh, ref1 a
// tetrahedron of the other volume.
public bool findVolumeVolumeFilter(DfFilterArray arr, uint volumeId0, uint volumeVertexId0,
                                   uint volumeId1, uint volumeTetId1)
{
    return dfFind(arr, PxgEncodeDeformableVertex(volumeId0, volumeVertexId0),
                  PxgEncodeDeformableTetrahedron(volumeId1, volumeTetId1));
}

// Rigid against volume: ref0 the rigid's filter reference, ref1 a tetrahedron.
public bool findRigidVolumeFilter(DfFilterArray arr, uint64_t rigidId, uint rigidCacheRef,
                                  uint volumeId, uint volumeTetId)
{
    return dfFind(arr, PxgEncodeRigidFilterRef(rigidId, rigidCacheRef),
                  PxgEncodeDeformableTetrahedron(volumeId, volumeTetId));
}

// Volume against surface, single-vertex paths: ref0 a tetrahedron, ref1 a vertex.
public bool findVolumeSurfaceVertexFilter(DfFilterArray arr, uint volumeId, uint volumeTetId,
                                          uint surfaceId, uint surfaceVertexId)
{
    return dfFind(arr, PxgEncodeDeformableTetrahedron(volumeId, volumeTetId),
                  PxgEncodeDeformableVertex(surfaceId, surfaceVertexId));
}

// Volume against surface, whole-triangle paths: ref0 a tetrahedron, ref1 a triangle.
public bool findVolumeSurfaceTriangleFilter(DfFilterArray arr, uint volumeId, uint volumeTetId,
                                            uint surfaceId, uint surfaceTriId)
{
    return dfFind(arr, PxgEncodeDeformableTetrahedron(volumeId, volumeTetId),
                  PxgEncodeDeformableTriangle(surfaceId, surfaceTriId));
}

// Rigid against surface, single-vertex paths: ref0 the rigid's filter reference, ref1 a vertex.
public bool findRigidSurfaceVertexFilter(DfFilterArray arr, uint64_t rigidId, uint rigidCacheRef,
                                         uint surfaceId, uint surfaceVertexId)
{
    return dfFind(arr, PxgEncodeRigidFilterRef(rigidId, rigidCacheRef),
                  PxgEncodeDeformableVertex(surfaceId, surfaceVertexId));
}

// Rigid against surface, whole-triangle paths: ref0 the rigid's filter reference, ref1 a triangle.
public bool findRigidSurfaceTriangleFilter(DfFilterArray arr, uint64_t rigidId, uint rigidCacheRef,
                                           uint surfaceId, uint surfaceTriId)
{
    return dfFind(arr, PxgEncodeRigidFilterRef(rigidId, rigidCacheRef),
                  PxgEncodeDeformableTriangle(surfaceId, surfaceTriId));
}

// Surface against surface, vertex-triangle: ref0 a vertex of the probing surface, ref1 a
// triangle of the other.
public bool findSurfaceSurfaceVertexTriangleFilter(DfFilterArray arr, uint surfaceId0, uint surfaceVertexId0,
                                                   uint surfaceId1, uint surfaceTriId1)
{
    return dfFind(arr, PxgEncodeDeformableVertex(surfaceId0, surfaceVertexId0),
                  PxgEncodeDeformableTriangle(surfaceId1, surfaceTriId1));
}

// Surface against surface, edge-edge: both refs are triangle edges.
public bool findSurfaceSurfaceEdgeEdgeFilter(DfFilterArray arr, uint surfaceId0, uint surfaceTriId0,
                                             uint localEdge0, uint surfaceId1, uint surfaceTriId1,
                                             uint localEdge1)
{
    return dfFind(arr, PxgEncodeDeformableTriangleEdge(surfaceId0, surfaceTriId0, localEdge0),
                  PxgEncodeDeformableTriangleEdge(surfaceId1, surfaceTriId1, localEdge1));
}
