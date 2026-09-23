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

// npTriTriWrite.slang -- P9.83b-5.
//
//   Fixture entries for the TAIL of writePatchesAndContacts
//   (convexMeshMidphase.cu:673-751):
//
//     ttrScan   -- ttpPatchStartIndices, the zero-padded block scan of :698-705
//     ttrWrite  -- ttpWritePatchAndContacts, the header write (:727-734) and the
//                  point write (:736-750) for ONE patch
//
// FULLY BDA, NO RWStructuredBuffer.  npTriTriAccept.slang records why that
// matters: Slang numbers RWStructuredBuffer bindings across the FILE, so adding a
// second entry with its own buffers silently shifts the first entry's bindings and
// the symptom is a segfault in an unrelated test.  A file that binds nothing
// cannot have that problem, and this one has to talk to raw streams anyway --
// the body under test writes through device pointers, not through a buffer view.
//
// WHAT THE FIXTURE HAS TO CONSTRUCT, over and above random cases:
//
//   * BOTH SENTINELS, SEPARATELY.  patchByteOffset = 0xFFFFFFFF with a live
//     contact offset, and the reverse.  The CUDA's two guards are independent
//     (:651 vs :746) and a case set where they always agree would let a port
//     merge them.
//   * isFlipped ON BOTH VALUES with the SAME material pair, so the swap at :729
//     is observable.  A fixture with mi0 == mi1 would see nothing.
//   * A NON-IDENTITY ROTATION WITH TRANSLATION.  The normal is rotated only
//     (:732) and the point is rotated AND translated (:745); with rotP = 0 the
//     two spellings are indistinguishable.
//   * pointCount = 0, so the header goes out with nbContacts = 0 and no points
//     follow -- the shape createNewPatches:955 can produce.
//   * A patchStartIndex that is NOT patchIndex * 6, because the scan is over
//     actual point counts and a port that recomputed it from the index would
//     agree on every full patch.
//
// No NoContraction stamp, for the reason in testNpCrsWrapper2.cpp.

import pxgSlangTriTriPatch;

struct TtwPush
{
    uint64_t cases;          // @0   TtwCase*
    uint64_t patchStream;    // @8
    uint64_t contactStream;  // @16
    uint64_t materials;      // @24  PxsMaterialDataGpu*
    uint64_t scanOut;        // @32  uint*, 132 per case
    uint     numCases;       // @40
    uint     _pad44;
}

[[vk::push_constant]]
ConstantBuffer<TtwPush> gTw;

// 704 bytes.  Laid out as float4/uint4 slots and read through Ptr<> at explicit
// offsets rather than as one struct: the counts tail is 128 uints and a struct
// member array under BDA layout is one more thing to get wrong for no gain.
static const uint TTW_CASE_STRIDE = 704u;
static const uint TTW_SCAN_STRIDE = 132u;   // uints: 128 starts + total + 3 pad

[shader("compute")]
[numthreads(64, 1, 1)]
void ttrScan(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gTw.numCases) return;

    const uint64_t base = gTw.cases + uint64_t(t) * uint64_t(TTW_CASE_STRIDE);
    const uint numPatchesToKeep = Ptr<uint4>(base + 176UL)[0].x;

    uint counts[TTP_MAX_MESH_MESH_PATCHES];
    Ptr<uint> src = Ptr<uint>(base + 192UL);
    for (uint i = 0; i < TTP_MAX_MESH_MESH_PATCHES; ++i)
        counts[i] = src[i];

    uint starts[TTP_MAX_MESH_MESH_PATCHES];
    const uint total = ttpPatchStartIndices(counts, numPatchesToKeep, starts);

    Ptr<uint> dst = Ptr<uint>(gTw.scanOut + uint64_t(t) * uint64_t(TTW_SCAN_STRIDE) * 4UL);
    for (uint i = 0; i < TTP_MAX_MESH_MESH_PATCHES; ++i)
        dst[i] = starts[i];
    dst[TTP_MAX_MESH_MESH_PATCHES] = total;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ttrWrite(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gTw.numCases) return;

    const uint64_t base = gTw.cases + uint64_t(t) * uint64_t(TTW_CASE_STRIDE);

    TtpPatch p;
    for (uint k = 0; k < TTP_MESH_MESH_CONTACT_LIMIT; ++k)
        p.keepPoints[k] = Ptr<float4>(base + uint64_t(k) * 16UL)[0];

    const float4 nrm = Ptr<float4>(base + 96UL)[0];
    p.normal = nrm.xyz;
    p.deepest = nrm.w;

    const float4 rotQ = Ptr<float4>(base + 112UL)[0];
    const float4 rotP = Ptr<float4>(base + 128UL)[0];
    const uint4 a = Ptr<uint4>(base + 144UL)[0];
    const uint4 b = Ptr<uint4>(base + 160UL)[0];

    p.pointCount = a.x;

    ttpWritePatchAndContacts(gTw.patchStream, b.x,
                             gTw.contactStream, b.y,
                             gTw.materials,
                             a.y, a.z, p,
                             rotQ, rotP.xyz, a.w != 0u,
                             b.z, b.w);
}
