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

// convexHullProbeBda.slang
//
// Test scaffolding for pxgSlangConvexHull — NOT an engine kernel and absent from
// the manifest.  One thread per hull; it reports every derived section address as
// a byte offset from the blob base, the three decoded counts, and one sampled
// element from every section, so the fixture can check the ADDRESSES and the
// VALUES independently.
//
// Output record, 128 bytes per hull:
//   @0   uint  offsets[7]   vertices, planes, polyDescs, verticesByEdges16,
//                           facesByEdges8, facesByVertices8, vertexData8
//   @28  uint  nbEdges, nbVerts, nbPolygons
//   @40  uint  polyDesc, vRef8, polyNbVerts, polyMinIndex   (for probe.polyIndex)
//   @56  uint  edgeV0, edgeV1, edgeF0, edgeF1               (for probe.edgeIndex)
//   @72  uint  vertF0, vertF1, vertF2, polyVertex           (for probe.vertIndex)
//   @96  float4 centerOfMass
//   @112 float4 vertex[probe.vertIndex]
//
// The two float4s sit at 96 and 112 and not packed straight after the uints at 88:
// a float4 store wants a 16-aligned address, and with a 128-byte record only
// multiples of 16 are 16-aligned for every hull in the array.

import pxgSlangConvexHull;

struct HullProbeParams
{
    uint64_t hulls;       // @0  array of blob base addresses (uint64 each)
    uint64_t output;      // @8
    uint     numHulls;    // @16
    uint     polyIndex;   // @20
    uint     edgeIndex;   // @24
    uint     vertIndex;   // @28 -> 32-byte blob
}

[[vk::push_constant]]
ConstantBuffer<HullProbeParams> gHp;

static const uint HP_OUT_STRIDE = 128u;

[shader("compute")]
[numthreads(64, 1, 1)]
void convexHullProbeLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint h = dtid.x;
    if (h >= gHp.numHulls)
        return;

    const uint64_t hull = Ptr<uint64_t>(gHp.hulls)[h];
    const uint64_t out = gHp.output + uint64_t(h) * uint64_t(HP_OUT_STRIDE);

    const uint desc = chHullDesc(hull);

    Ptr<uint>(out)[0] = uint(chVertices(hull) - hull);
    Ptr<uint>(out)[1] = uint(chPlanes(hull, desc) - hull);
    Ptr<uint>(out)[2] = uint(chPolyDescs(hull, desc) - hull);
    Ptr<uint>(out)[3] = uint(chVerticesByEdges16(hull, desc) - hull);
    Ptr<uint>(out)[4] = uint(chFacesByEdges8(hull, desc) - hull);
    Ptr<uint>(out)[5] = uint(chFacesByVertices8(hull, desc) - hull);
    Ptr<uint>(out)[6] = uint(chVertexData8(hull, desc) - hull);

    Ptr<uint>(out)[7] = chNbEdges(desc);
    Ptr<uint>(out)[8] = chNbVerts(desc);
    Ptr<uint>(out)[9] = chNbPolygons(desc);

    const uint pd = chPolyDesc(hull, desc, gHp.polyIndex);
    Ptr<uint>(out)[10] = pd;
    Ptr<uint>(out)[11] = chPolyVRef8(pd);
    Ptr<uint>(out)[12] = chPolyNbVerts(pd);
    Ptr<uint>(out)[13] = chPolyMinIndex(pd);

    Ptr<uint>(out)[14] = chEdgeVertex(hull, desc, gHp.edgeIndex, 0u);
    Ptr<uint>(out)[15] = chEdgeVertex(hull, desc, gHp.edgeIndex, 1u);
    Ptr<uint>(out)[16] = chEdgeFace(hull, desc, gHp.edgeIndex, 0u);
    Ptr<uint>(out)[17] = chEdgeFace(hull, desc, gHp.edgeIndex, 1u);

    Ptr<uint>(out)[18] = chVertexFace(hull, desc, gHp.vertIndex, 0u);
    Ptr<uint>(out)[19] = chVertexFace(hull, desc, gHp.vertIndex, 1u);
    Ptr<uint>(out)[20] = chVertexFace(hull, desc, gHp.vertIndex, 2u);
    Ptr<uint>(out)[21] = chPolyVertex(hull, desc, chPolyVRef8(pd), 0u);

    Ptr<float4>(out + 96UL)[0] = chCenterOfMass(hull);
    Ptr<float4>(out + 112UL)[0] = chVertex(hull, gHp.vertIndex);
}
