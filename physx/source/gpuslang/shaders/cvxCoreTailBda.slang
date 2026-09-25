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

// cvxCoreTailBda.slang -- P9.87.
//
//   Fixture entries that run the two convexCore emission tails THEMSELVES, on
//   raw device streams:
//
//     cctSingle -- cvxCoreWriteOutput   (pxgSlangCvxCoreOut.slang), the
//                  TestOutput::writeOutput PxContact-array overload
//                  (convexCoreCollision.cu:177-290 + insertIntoPatchStream,
//                  contactPatchUtils.cuh:139-180) that the plane and convex
//                  kernels end with
//     cctMulti  -- cvxCoreWriteOutput32 (pxgSlangCvxCoreOut32.slang), the
//                  Gu::Contact overload (:292-445) that the trimesh kernel ends
//                  with
//
// WHY THIS EXISTS.  Until P9.87 nothing but the scenes ran either tail, and a scene
// sees rest heights, not bytes.  npCvxCoreOut (ccoWriteContacts/ccoWriteContact) is
// a separate transcription into a zero-initialised RECORD, so it cannot tell "left
// the stream's byte alone" from "wrote 0" -- which is exactly the difference P9.87
// is about: PxContactPatch is written field by field, bytes 54-63 are never stored,
// and the engine's patch stream is never cleared.  Here every buffer the tails
// touch starts as per-word poison and is compared whole.
//
// THE SAME TEXT THE ENGINE COMPILES.  Both tails are #included, as the engine
// kernels include them (cvxCorePlaneNpBda.slang, cvxCoreCvxNpBda.slang,
// cvxCoreTriNpBda.slang), so what is verified is the engine's instructions and not
// a copy of them.  pxgSlangCvxCoreAcc.slang is included only for MAX_PATCHES /
// MAX_PATCH_POINTS, which cvxCoreWriteOutput32 takes its array bounds from; without
// PXG_CVXCORE_ACC_HAS_HELPERS, as npCvxCoreWarpAcc.slang includes it, because
// pxgSlangCvxCoreShape is not imported here.
//
// FULLY BDA, NO RWStructuredBuffer, for the reason npTriTriWrite.slang gives: two
// entries in one file with bound buffers shift each other's bindings.
//
// PER-CASE COUNTERS.  Each case gets its own PxgPatchAndContactCounters (16 B at
// counters + t*16), seeded by the host so the case owns a disjoint slice of each
// stream.  The InterlockedAdds are then deterministic, and the whole-buffer image
// the mirror builds can be compared word for word.
//
// STAMPED NoContraction, as every engine importer of these tails is.  Nothing here
// multiplies and adds in one expression except the material combine, and that is
// the engine's text.

import pxgSlangNpAbi;

#include "pxgSlangCvxCoreOut.slang"
#include "pxgSlangCvxCoreAcc.slang"
#include "pxgSlangCvxCoreOut32.slang"

struct CctPush
{
    uint64_t cases;          // @0   CctCase*, CCT_CASE_STRIDE bytes each
    uint64_t cmOutputs;      // @8   PxsContactManagerOutput[numCases]
    uint64_t materials;      // @16  PxsMaterialDataGpu*
    uint64_t contactStream;  // @24
    uint64_t patchStream;    // @32
    uint64_t counters;       // @40  PxgPatchAndContactCounters[numCases]
    uint64_t touchFlags;     // @48  uint[numCases]
    uint64_t patchFlags;     // @56  uint[numCases]
    uint64_t startPatches;   // @64  the start* bases the output pointers are built from
    uint64_t startPoints;    // @72
    uint64_t startForces;    // @80
    uint     numCases;       // @88
    uint     _pad92;
}

[[vk::push_constant]]
ConstantBuffer<CctPush> gCt;

// The case record, 1376 bytes, read through Ptr<> at explicit offsets:
//   @0     uint4  hdr   x = nbContacts (single) / numPatches (multi), y = numPoints (multi),
//                       z = shape0.materialIndex, w = shape1.materialIndex (full u32)
//   @16    uint4  lim   x = patchBytesLimit, y = contactBytesLimit, z = forceBytesLimit
//   @32    uint4  patchCount[4]    16 counts
//   @96    float4 patchNormal[16]  xyz
//   @352   float4 points[64]       xyz = point, w = separation
static const uint CCT_CASE_STRIDE = 1376u;

PxgShapeGpu cctShape(uint materialIndex)
{
    PxgShapeGpu s;
    s.scaleX = 0.0f; s.scaleY = 0.0f; s.scaleZ = 0.0f;
    s.scaleRotX = 0.0f; s.scaleRotY = 0.0f; s.scaleRotZ = 0.0f; s.scaleRotW = 0.0f;
    s.materialIndex = materialIndex;
    s.hullOrMeshPtr = 0uL;
    s.type = 0u;
    s.particleOrSoftbodyId = 0u;
    return s;
}

CvxCoreOutRefs cctRefs(uint t, uint4 lim)
{
    CvxCoreOutRefs o;
    o.cmOutputs = gCt.cmOutputs;
    o.materials = gCt.materials;
    o.contactStream = gCt.contactStream;
    o.patchStream = gCt.patchStream;
    o.patchAndContactCounters = gCt.counters + uint64_t(t) * 16uL;
    o.touchChangeFlags = gCt.touchFlags;
    o.patchChangeFlags = gCt.patchFlags;
    o.startContactPatches = gCt.startPatches;
    o.startContactPoints = gCt.startPoints;
    o.startContactForces = gCt.startForces;
    o.patchBytesLimit = lim.x;
    o.contactBytesLimit = lim.y;
    o.forceBytesLimit = lim.z;
    return o;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cctSingle(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gCt.numCases) return;

    const uint64_t base = gCt.cases + uint64_t(t) * uint64_t(CCT_CASE_STRIDE);
    const uint4 hdr = Ptr<uint4>(base)[0];
    const uint4 lim = Ptr<uint4>(base + 16UL)[0];
    const float3 normal = Ptr<float4>(base + 96UL)[0].xyz;

    float3 pointsOut[4];
    float distsOut[4];
    for (uint i = 0u; i < 4u; ++i)
    {
        const float4 p = Ptr<float4>(base + 352UL + uint64_t(i) * 16UL)[0];
        pointsOut[i] = p.xyz;
        distsOut[i] = p.w;
    }

    cvxCoreWriteOutput(t, cctShape(hdr.z), cctShape(hdr.w), normal, pointsOut, distsOut, hdr.x,
                       cctRefs(t, lim));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cctMulti(uint3 dtid : SV_DispatchThreadID)
{
    const uint t = dtid.x;
    if (t >= gCt.numCases) return;

    const uint64_t base = gCt.cases + uint64_t(t) * uint64_t(CCT_CASE_STRIDE);
    const uint4 hdr = Ptr<uint4>(base)[0];
    const uint4 lim = Ptr<uint4>(base + 16UL)[0];

    CvxCoreContact32 c;
    c.numPatches = hdr.x;
    c.numPoints = hdr.y;
    for (uint i = 0u; i < MAX_PATCHES; ++i)
    {
        c.patchCount[i] = Ptr<uint>(base + 32UL)[i];
        c.patchNormal[i] = Ptr<float4>(base + 96UL + uint64_t(i) * 16UL)[0].xyz;
    }
    for (uint i = 0u; i < MAX_PATCHES * MAX_PATCH_POINTS; ++i)
        c.points[i] = Ptr<float4>(base + 352UL + uint64_t(i) * 16UL)[0];

    cvxCoreWriteOutput32(t, cctShape(hdr.z), cctShape(hdr.w), c, cctRefs(t, lim));
}
