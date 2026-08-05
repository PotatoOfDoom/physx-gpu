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

// pinnedAtomicBda.slang
//
// Test scaffolding — NOT an engine kernel, absent from the manifest.
//
// The convex-vs-trimesh core kernel's prologue does an atomicMax on
// `maxTempMemRequirement` (convexMesh.cu:674), a word that PhysX allocates in
// PINNED HOST MEMORY and reads on the CPU with no copy (PxgNarrowphaseCore.cpp:223
// and :1644).  p3ShimPinned already proves that a pinned allocation is
// device-writable and host-coherent through its mapping — but only for ordinary
// STORES.  An atomic read-modify-write on host-visible memory is a separate
// promise, and this shader is what makes it a checked one rather than an assumed
// one.
//
// Every invocation contends on the same word, so the result is the maximum over
// the whole dispatch and any lost update shows up as a low value rather than as a
// crash.

struct PinnedAtomicParams
{
    uint64_t target;   // @0  the word to maximise into
    uint64_t counter;  // @8  a second word, incremented once per invocation
    uint     count;    // @16 invocations that should contribute
    uint     _pad20;
}

[[vk::push_constant]]
ConstantBuffer<PinnedAtomicParams> gPa;

[shader("compute")]
[numthreads(64, 1, 1)]
void pinnedAtomicLaunch(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= gPa.count)
        return;

    // A deliberately non-monotonic sequence, so a maximum that merely tracks the
    // last writer would land somewhere other than the true maximum.
    const uint v = ((i * 2654435761u) >> 8) & 0xffffu;

    uint old;
    InterlockedMax(Ptr<uint>(gPa.target)[0], v, old);
    InterlockedAdd(Ptr<uint>(gPa.counter)[0], 1u, old);
}
