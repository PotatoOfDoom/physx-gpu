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

// pxgSlangTex3d.slang -- P9.12a, the CUtexObject replacement.
//
// WHY THIS EXISTS.  The engine's SDF path stores a texture handle INSIDE a
// device-side struct and dereferences it in the kernel:
//
//     struct PxgTriangleMesh { ...; CUtexObject mTexObject; ... };
//     tex3D<float>(mesh.mTexObject, x, y, z);          // sdfCollision.cuh:113
//
// SPIR-V has no texture handle that can live in a buffer.  The equivalent is
// DESCRIPTOR INDEXING: one unbounded array of sampled 3D images bound once per
// device, with the handle as the ARRAY INDEX.  A CUtexObject therefore ports to
// a plain uint, storable and loadable exactly as the CUDA stores it, and
// `mTexObject == 0` keeps meaning "no texture" because slot 0 is never filled
// (PxgGeometryManager.cpp:725-731 leaves it at 0 for a mesh with no SDF).
//
// THREE THINGS THAT ARE NOT OPTIONAL, each of them a difference between the
// CUDA and the Vulkan that a transcription would get silently wrong:
//
//  T1. CUDA's tex3D takes UNNORMALIZED coordinates -- cuTexObjectCreate is
//      called without CU_TRSF_NORMALIZED_COORDINATES, so `f.x + 0.5f` is a
//      texel-space coordinate.  Vulkan's unnormalizedCoordinates sampler flag
//      is FORBIDDEN on 3D images, so the coordinate must be divided by the
//      extent here.  That is why every entry point below queries the extent
//      rather than taking it as an argument: the sparse subgrid texture's
//      extent is computed in PxgGeometryManager and stored NOWHERE the kernel
//      can see (PxgTriangleMesh carries sdfDims, which is the DENSE grid).
//  T2. The integer formats are read as NORMALIZED floats.  cuTexObjectCreate is
//      called without CU_TRSF_READ_AS_INTEGER, so an 8- or 16-bit subgrid
//      sample arrives in [0,1] and SparseSDFTexture::applySubgridSdfScale
//      rescales it.  The Vulkan equivalents are therefore the UNORM formats,
//      and the shader side needs no branch at all -- the sampler returns the
//      same [0,1] float for all three.
//  T3. Filtering is NOT bit-exact against CUDA.  CUDA's linear filter uses
//      9-bit fixed-point weights; Vulkan's precision is implementation-defined.
//      Sampling at a texel CENTRE ((p + 0.5) / dim) is exact on both because
//      the weights are 0 and 1; anything between texels carries the
//      implementation's filtering precision.  Every fixture over this module
//      must therefore compare interior points against a TOLERANCE and pin the
//      exactness claim to texel centres -- the same arrangement npTex3d
//      established in P2.
//
// The set index is 1.  Set 0 stays the pipeline's own bindings, so a kernel
// that samples still marshals all of its buffers through device addresses as
// every other engine-ABI kernel does; the two mechanisms are independent and
// the SPIR-V carries PhysicalStorageBufferAddresses and RuntimeDescriptorArray
// side by side.

[[vk::binding(0, 1)]] Texture3D<float> gPxgTex3D[];
[[vk::binding(1, 1)]] SamplerState gPxgTex3DSampler;

// The extent of a bindless volume, as the CUDA never has to ask because a
// CUarray carries its own descriptor.
uint3 pxgTex3DSize(uint handle)
{
    uint w, h, d;
    gPxgTex3D[NonUniformResourceIndex(handle)].GetDimensions(w, h, d);
    return uint3(w, h, d);
}

// CUDA's `tex3D<float>(texture, x, y, z)`, coordinate convention included:
// UNNORMALIZED texel space, clamp-to-edge, trilinear (T1/T3).
//
// NonUniformResourceIndex is load-bearing rather than decorative.  The handle
// is read out of memory, so lanes in one subgroup routinely carry DIFFERENT
// handles -- two contact-manager pairs against two different meshes land in the
// same wave.  Without the decoration the driver may sample every lane through
// the first lane's descriptor, which is not a crash but a wrong number.
float pxgTex3D(uint handle, float3 xyz)
{
    const uint3 dim = pxgTex3DSize(handle);
    const float3 uvw = xyz / float3(float(dim.x), float(dim.y), float(dim.z));
    return gPxgTex3D[NonUniformResourceIndex(handle)].SampleLevel(gPxgTex3DSampler, uvw, 0.0f);
}

float pxgTex3D(uint handle, float x, float y, float z)
{
    return pxgTex3D(handle, float3(x, y, z));
}
