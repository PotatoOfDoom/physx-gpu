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

// sphereHfCoreBda.slang
//
// Test scaffolding for pxgSlangSphereHfCore -- NOT an engine kernel, absent from
// the manifest.  P9.52a.
//
// ONE THREAD PER PAIR, which is the shape the CUDA core itself has: unlike the
// convex path (one warp per pair, with a SAT and a clipper spread over 32 lanes),
// `sphereHeightfieldNarrowphaseCore` is entirely scalar and its enclosing kernel
// hands each thread its own pair (convexHeightfield.cu:874).  Emulating a warp
// here would test something the core does not do.
//
// The four record arrays are REAL BUFFERS at their engine strides, so the fixture
// checks where each write landed and not merely what it computed.  They are what
// the engine kernel will reach through `maCompute`; the offsets come from the
// ConvexMeshPair blob exactly as they will there.
//
// Output record, 32 bytes:
//   @0   uint  nbOutputContacts
//   @4   uint  outMask
//   @8   uint  pairOffset
//   @12  uint  _pad
//   @16  float4 outContact.xyz, outSep

import pxgSlangSphereHfCore;

struct SphereHfCoreParams
{
    uint64_t cmInputs;          // @0
    uint64_t transformCache;    // @8
    uint64_t contactDistance;   // @16
    uint64_t gpuShapes;         // @24
    uint64_t sphereTrimeshPair; // @32
    uint64_t ni;                // @40
    uint64_t maxDepth;          // @48
    uint64_t interm;            // @56
    uint64_t ordered;           // @64
    uint64_t pairs;             // @72
    uint64_t output;            // @80
    uint     numPairs;          // @88
    uint     _pad92;
}

[[vk::push_constant]]
ConstantBuffer<SphereHfCoreParams> gShc;

static const uint SHC_OUT_STRIDE = 32u;

[shader("compute")]
[numthreads(64, 1, 1)]
void sphereHfCoreLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint p = dtid.x;
    if (p >= gShc.numPairs)
        return;

    const uint4 curPair = Ptr<uint4>(gShc.pairs)[p];

    // `globalThreadIndex` is the pair's index in the whole dispatch, which is what
    // the CUDA kernel passes (:874) and what the deferred record must carry so the
    // second pass can re-derive the pair.  It is a SEPARATE argument there, not a
    // field of the pair record; the midphase fills .w with `shapeRef1`
    // (convexHfMidphaseBda:505) and the core never reads it, so the probe reuses
    // that slot to drive the parameter with a value of the fixture's choosing.
    // Passing `p` instead would make the two indistinguishable, and the deferred
    // word is `globalThreadIndex | mask` -- exactly where the difference shows.
    const ShCoreOut o = shSphereHfCore(curPair.w,
                                       gShc.cmInputs, gShc.transformCache,
                                       gShc.contactDistance, gShc.gpuShapes,
                                       gShc.sphereTrimeshPair,
                                       gShc.ni, gShc.maxDepth, gShc.interm, gShc.ordered,
                                       curPair);

    const uint64_t out = gShc.output + uint64_t(p) * uint64_t(SHC_OUT_STRIDE);
    Ptr<uint>(out)[0] = o.nbOutputContacts;
    Ptr<uint>(out)[1] = o.outMask;
    Ptr<uint>(out)[2] = o.pairOffset;
    Ptr<uint>(out)[3] = 0u;
    Ptr<float4>(out + 16UL)[0] = float4(o.outContact, o.outSep);
}
