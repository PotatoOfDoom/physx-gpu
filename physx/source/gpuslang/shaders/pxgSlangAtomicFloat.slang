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

// pxgSlangAtomicFloat — CUDA's atomicAdd(float*) on a buffer-device address.
//
// WHY THIS MODULE EXISTS.  Several deformable kernels accumulate a per-vertex
// position correction that many threads contribute to at once: the CUDA original
// uses atomicAdd, and the Slang ports originally lowered it to a plain
// load-add-store.  Each of those ports documented the same validity condition --
// "no two contacts share a vertex within one pass, so there is no cross-thread
// contention, and the read-modify-write is bit-identical to the atomic".
//
// THAT CONDITION IS FALSE IN EVERY REAL SCENE.  A cloth triangle contact bumps the
// reference count of all THREE of its vertices, and adjacent triangles share
// vertices by construction; a folded cloth's self-collision has many contacts per
// vertex.  Under contention a non-atomic RMW silently drops updates, and because
// mDeltaPos.w is the DIVISOR that applyExternalDeltas uses to mass-average
// (scale = 1/max(w,1)), losing count updates desynchronises the accumulated .xyz
// from its divisor -- the correction is then applied with the wrong weight, which
// looks like an unstable or under-responsive solve rather than like a lost write.
//
// This was found when a grid-coverage fix DOUBLED the self-collision contact count
// in the clothselfcollide scene (VT 76 -> 152, EE 34 -> 70, measured) and the
// scene got markedly WORSE instead of better: the extra contention, not the extra
// contacts, was what broke it.
//
// LOWERING.  Slang's InterlockedAdd on a Ptr<float> emits OpAtomicFAddEXT with the
// AtomicFloat32AddEXT capability and the SPV_EXT_shader_atomic_float_add
// extension.  The bridge enables VK_EXT_shader_atomic_float +
// shaderBufferFloat32AtomicAdd where the device advertises them
// (PxgVulkanComputeDevice.cpp); llvmpipe does.  A shader that uses these helpers
// will fail PIPELINE CREATION on a device without the feature -- there is no
// silent fallback, deliberately, because a silent fallback is exactly the bug this
// module exists to remove.
//
// DETERMINISM.  Float addition is not associative, so an atomic accumulation is
// order-dependent and therefore not bit-reproducible run to run.  That is equally
// true of the CUDA original -- atomicAdd(float*) has the same property -- so this
// is parity with the .cu, not a regression.  The non-atomic RMW it replaces was
// not deterministic either; it was merely wrong.
module pxgSlangAtomicFloat;

// atomicAdd(&p[i], v) for a float array at a device address.
public void pxgAtomicAddF32(uint64_t base, uint index, float v)
{
    float prev;
    InterlockedAdd((Ptr<float>(base))[index], v, prev);
}

// The float4-lane forms: a float4 array at `base`, element `index`, one component.
// Component addresses are base + index*16 + {0,4,8,12}; each is its own atomic, as
// in the CUDA original (three separate atomicAdds for .x/.y/.z).
public void pxgAtomicAddF4X(uint64_t base, uint index, float v)
{
    pxgAtomicAddF32(base + uint64_t(index) * 16UL, 0u, v);
}
public void pxgAtomicAddF4Y(uint64_t base, uint index, float v)
{
    pxgAtomicAddF32(base + uint64_t(index) * 16UL + 4UL, 0u, v);
}
public void pxgAtomicAddF4Z(uint64_t base, uint index, float v)
{
    pxgAtomicAddF32(base + uint64_t(index) * 16UL + 8UL, 0u, v);
}
public void pxgAtomicAddF4W(uint64_t base, uint index, float v)
{
    pxgAtomicAddF32(base + uint64_t(index) * 16UL + 12UL, 0u, v);
}

// The whole xyz vector, three atomics — the shape every cloth/softbody delta
// scatter needs.
public void pxgAtomicAddF4XYZ(uint64_t base, uint index, float3 v)
{
    pxgAtomicAddF4X(base, index, v.x);
    pxgAtomicAddF4Y(base, index, v.y);
    pxgAtomicAddF4Z(base, index, v.z);
}
