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

// pxgSlangTex3d.slang -- P9.12a, the CUtexObject replacement; P9.96, the filter
// emulated from texel loads.
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
//      texel-space coordinate.  Until P9.96 this module handed a Vulkan sampler
//      the coordinate divided by the extent (Vulkan forbids unnormalized
//      coordinates on 3D images), and that round trip is not exact on a
//      non-power-of-two extent (P9.95e: 85 of 1024 texel-centre points of a
//      10 x 7 x 11 grid off by an ulp or more).  Since P9.96 there is no sampler
//      and no division: the filter below is evaluated in texel space from eight
//      texel LOADS.  The extent is still QUERIED, now for the clamp, because the
//      sparse subgrid texture's extent is computed in PxgGeometryManager and
//      stored NOWHERE the kernel can see (PxgTriangleMesh carries sdfDims, which
//      is the DENSE grid).  It is queried ONCE per sample and passed down.
//  T2. The integer formats are read as NORMALIZED floats.  cuTexObjectCreate is
//      called without CU_TRSF_READ_AS_INTEGER, so an 8- or 16-bit subgrid
//      sample arrives in [0,1] and SparseSDFTexture::applySubgridSdfScale
//      rescales it.  The Vulkan equivalents are the UNORM formats and the shader
//      needs no branch: Load returns the same [0,1] float for all three.  WHICH
//      float is the driver's.  Mesa's llvmpipe converts with
//      lp_build_unsigned_norm_to_float (gallivm lp_bld_conv.c: SIToFP, then an
//      FMul by the constant float(1.0/(2^b - 1))), reached from the fetch path
//      through lp_build_fetch_rgba_soa (lp_bld_format_soa.c) -- read from Mesa
//      main, the installed driver is 25.2.8.  That is bit-for-bit the engine's
//      own spelling, v * (1.0f/255.0f) and v * (1.0f/65535.0f)
//      (Gu::SDF::decodeSample, GuSDF.h:167/:171), and NOT the correctly rounded
//      quotient v/255 (they differ on 126 of 256 and 512 of 65536 codes).  The
//      source reading is a prediction; the MEASUREMENT is the fixture
//      npTex3dBindlessUnormExhaustive, which reads every 8-bit and every 16-bit
//      code back through Load and through pxgTex3D and requires the engine's
//      spelling bit for bit (P9.96 outcome A).
//  T3. THE FILTER IS EMULATED, and every fixture over this module is bitwise
//      against a -ffp-contract=off host oracle (tests/pxgTex3dRef.h) rather than
//      held to a tolerance.  The reading is the CUDA Programming Guide's
//      "Texture Fetching / Linear Filtering" (Appendix D in Guide 2.0):
//          xB = x - 0.5,  i = floor(xB),  alpha = frac(xB)    (same for y, z)
//          tex(x,y,z) = (1-a)(1-b)(1-g) T[i,j,k]     + a(1-b)(1-g) T[i+1,j,k]
//                     + (1-a)b(1-g)     T[i,j+1,k]   + ab(1-g)     T[i+1,j+1,k]
//                     + (1-a)(1-b)g     T[i,j,k+1]   + a(1-b)g     T[i+1,j,k+1]
//                     + (1-a)bg         T[i,j+1,k+1] + abg         T[i+1,j+1,k+1]
//      with a, b, g "stored in 9-bit fixed point format with 8 bits of
//      fractional value".  In fp32 every step up to the weights is EXACT:
//      xB (the coordinate is clamped to >= 0.5 first), floor, frac, the
//      quantization below, (1 - a), and the weight products (three factors
//      k/256, numerators <= 2^24).  Only the eight texel products and the seven
//      adds round, and NoContraction (tests/CMakeLists.txt, PXG_NOCONTRACT_SHADERS)
//      keeps them unfused in every importer.  Filtering is applied AFTER the
//      UNORM conversion, as the formula is written over the fetched T.
//
// KNOWN UNKNOWNS.  This is an IEEE reading of the guide plus ONE third-party
// hardware measurement; it is not claimed to be bit-equal to tex3D on an NVIDIA
// GPU, and nothing in this repo can prove it is.  Documented, not reported --
// CUDA is underspecified here, not wrong:
//  K0. The claim is "the guide's formula, read in IEEE fp32", nothing more.
//  K1. WEIGHT ROUNDING: nearest 1/256, ties UP.  The one undocumented choice.
//      Evidence: a single third-party bit-level measurement on one architecture
//      (pantheonsim PR #145, RTX 3060 / sm_86: "rounds to the nearest 1/256,
//      halves up").  The guide's 9-bit remark does NOT discriminate rounding from
//      truncation -- 1 - a needs the value 1.0 under either.  D3D's 16.8
//      coordinate snapping supports rounding but uses ties-to-even.
//  K2. BLEND ARITHMETIC: the eight-term sum above in fp32, summed left to right
//      in the guide's printed term order.  That order is a TRANSCRIPTION
//      CONVENTION -- the guide gives a real-number formula with no evaluation
//      order.  Hardware is reported (same source) to form integer weights and
//      round the exact weighted sum once, and to filter UNORM texels as 16-bit
//      integers; neither is reproducible in fp32 nor documented by NVIDIA.
//  K3. UNORM CONVERSION: float(v) * (1/(2^b - 1)), see T2.  CUDA documents only
//      "normalized float in [0,1]".
//  K4. CLAMP: the COORDINATE is clamped to [0.5, N - 0.5] per axis, which is the
//      rule the same hardware measurement reports.  The guide's own text clamps
//      x to [0, N-1] ("x is replaced by 0 if x < 0 and N-1 if N <= x") and is
//      then incomplete at the edges (the formula still needs T[-1] or T[N]).
//      Completed with the index clamp below, that reading DISAGREES with this
//      one over the last half texel of every axis: for x > N - 1 it pins alpha
//      at 0.5 (the values part from x = N - 1 + 1/512, where the 1/256 weight
//      first moves), and for x >= N - 0.5 it gives (T[N-2] + T[N-1])/2 where
//      this rule gives T[N-1].  (Below x = 0.5 the two differ only by the
//      rounding of T[0] blended with itself.)  That band IS inside the engine
//      domain: the dense path clamps f to the box shrunk by the 0.5 margin, so
//      f + 0.5 reaches N - 0.5 for every point on, or clamped onto, a max face,
//      and the sparse coarse texture and the subgrid blocks stored last along an
//      axis are sampled up to their far faces too.  So the choice IS observable
//      in the engine, resting on that one measurement -- testSdfTexture's bite
//      assert counts the guide-text clamp (cc-guide) changing 539 of its
//      engine-domain expectations.  Only a rule with no coordinate clamp at all
//      (index clamp only) agrees with this one on every engine-domain value,
//      because the engine never samples outside [0.5, N - 0.5].  The INDEX clamp
//      kept after it is for ADDRESS SAFETY only: for non-NaN input it is a no-op
//      on i and moves only the weight-0 tap i + 1 = N onto T[N-1], which a zero
//      weight cannot see except through the SIGN of a zero result (K5).
//  K5. SPECIAL VALUES are what the formula gives and are not patched.  At a
//      texel centre the result is exactly T except for the SIGN of a zero: the
//      sum starts from 1 * T and adds +0 * t for every other tap (-0 for a
//      negative t, and -0 + +0 = +0), so a +0 texel returns +0, and a -0 texel
//      returns -0 iff every zero-weight tap has its sign bit set (on a far face
//      the clamped tap is T itself).  An inf or NaN zero-weight tap makes the
//      result NaN (0 * inf).  An INFINITE coordinate is clamped like any other
//      (GLSL.std.450 FMin/FMax are defined for +-inf) and returns the edge
//      sample; only a NaN COORDINATE gives unspecified (but in-range) indices and
//      an unspecified value -- never an out-of-bounds address.  SDF texels are
//      finite, so of all this only the sign rule can reach the engine (a -0
//      texel).  npTex3dBindless pins the sign rule and the infinite coordinates.
//
// SAFETY.  pxgTex3D clamps every texel index to [0, max(N-1, 0)] against the
// queried extent, so through it a VALID descriptor is never addressed out of
// bounds, whatever the coordinate.  pxgTex3DTexel, the raw fetch under it, clamps
// NOTHING: a direct caller owns its bounds (an out-of-bounds OpImageFetch is
// undefined without robustImageAccess).  Descriptor validity is a pre-existing
// exposure, unchanged in kind: the set is PARTIALLY_BOUND without nullDescriptor,
// so an ACTIVE lane that reaches pxgTex3D with handle 0, or with a slot that was
// never written or whose texture was destroyed (destroyTexture3D leaves the slot
// naming the destroyed view), is undefined.  Engine callers sample only a mesh
// for which sdfReadMesh reports an SDF, whose handles the engine wrote.  An
// INACTIVE lane performs no access (llvmpipe lowers each NonUniform access to a
// loop over the active lanes' handles).  The count of descriptor accesses per
// sample went from two (size query + sample) to nine (size query + eight loads).
// check_speculated_loads.py covers BDA loads, not image fetches; these are
// clamped by construction and reviewed here.
//
// The set index is 1.  Set 0 stays the pipeline's own bindings, so a kernel
// that samples still marshals all of its buffers through device addresses as
// every other engine-ABI kernel does; the two mechanisms are independent and
// the SPIR-V carries PhysicalStorageBufferAddresses and RuntimeDescriptorArray
// side by side.

[[vk::binding(0, 1)]] Texture3D<float> gPxgTex3D[];
// The set has NO other binding: until P9.96 a SamplerState sat at binding 1, and
// P9.97 removed it from the device's layout too.  Do not declare anything else in
// set 1 -- createPipelineBindless refuses such a module (pxgSpirvFitsBindlessLayout).

// 2^8: CUDA's 8 fractional weight bits.  STEP is 2^-8, exact, and a multiply --
// never a runtime division.
static const float PXG_TEX3D_WEIGHT_ONE = 256.0f;
static const float PXG_TEX3D_WEIGHT_STEP = 1.0f / PXG_TEX3D_WEIGHT_ONE;

// The extent of a bindless volume, as the CUDA never has to ask because a
// CUarray carries its own descriptor.
uint3 pxgTex3DSize(uint handle)
{
    uint w, h, d;
    gPxgTex3D[NonUniformResourceIndex(handle)].GetDimensions(w, h, d);
    return uint3(w, h, d);
}

// The 9-bit fixed-point weight (1.8): frac rounded to the nearest 1/256, ties UP
// (K1).  s = a * 2^8, q = floor(s) and r = s - q are all exact in fp32, so this
// is the exact real-number floor(a * 256 + 0.5) / 256 for every a in [0, 1) --
// brute-forced over all 1,065,353,216 such floats.  (The shorter spelling
// floor(a * 256 + 0.5) misrounds exactly one of them, 0x3affffff, which a frac
// computed from an fp32 coordinate never produces; this one does not depend on
// that.)
float pxgTex3DWeight(float a)
{
    const float s = a * PXG_TEX3D_WEIGHT_ONE;
    const float q = floor(s);
    const float r = s - q;
    return (q + (r >= 0.5f ? 1.0f : 0.0f)) * PXG_TEX3D_WEIGHT_STEP;
}

// Address safety only (K4): the int max() keeps a zero extent at index 0 rather
// than an upper bound of -1.
int pxgTex3DClampIdx(int i, uint n)
{
    return clamp(i, 0, max(int(n) - 1, 0));
}

// One texel through the bindless array.  NonUniformResourceIndex is on EVERY
// descriptor access, which is why this helper takes the HANDLE and not a cached
// Texture3D local: the decoration must reach each OpImageFetch.  It is
// load-bearing rather than decorative -- the handle is read out of memory, so
// lanes in one subgroup routinely carry DIFFERENT handles (two contact-manager
// pairs against two different meshes land in the same wave), and without it the
// driver may fetch every lane through the first lane's descriptor.
// NO BOUNDS CLAMP: the caller guarantees 0 <= x,y,z < the extent (pxgTex3D does it
// with pxgTex3DClampIdx; see SAFETY).
float pxgTex3DTexel(uint handle, int x, int y, int z)
{
    return gPxgTex3D[NonUniformResourceIndex(handle)].Load(int4(x, y, z, 0));
}

// CUDA's `tex3D<float>(texture, x, y, z)` for the engine's descriptor:
// UNNORMALIZED texel space, clamp on all three axes, linear filter (T1-T3, K0-K5).
float pxgTex3D(uint handle, float3 xyz)
{
    const uint3 dim = pxgTex3DSize(handle);

    // K4: the coordinate clamp.  float(n) - 0.5f is exact for n < 2^23.
    const float xc = min(max(xyz.x, 0.5f), float(dim.x) - 0.5f);
    const float yc = min(max(xyz.y, 0.5f), float(dim.y) - 0.5f);
    const float zc = min(max(xyz.z, 0.5f), float(dim.z) - 0.5f);

    // Exact: xc >= 0.5, so xB >= 0, and floor and frac of it are exact.
    const float xB = xc - 0.5f;
    const float yB = yc - 0.5f;
    const float zB = zc - 0.5f;
    const float fx = floor(xB);
    const float fy = floor(yB);
    const float fz = floor(zB);
    const float a = pxgTex3DWeight(xB - fx);
    const float b = pxgTex3DWeight(yB - fy);
    const float g = pxgTex3DWeight(zB - fz);

    const int i0 = pxgTex3DClampIdx(int(fx), dim.x);
    const int i1 = pxgTex3DClampIdx(int(fx) + 1, dim.x);
    const int j0 = pxgTex3DClampIdx(int(fy), dim.y);
    const int j1 = pxgTex3DClampIdx(int(fy) + 1, dim.y);
    const int k0 = pxgTex3DClampIdx(int(fz), dim.z);
    const int k1 = pxgTex3DClampIdx(int(fz) + 1, dim.z);

    const float t000 = pxgTex3DTexel(handle, i0, j0, k0);
    const float t100 = pxgTex3DTexel(handle, i1, j0, k0);
    const float t010 = pxgTex3DTexel(handle, i0, j1, k0);
    const float t110 = pxgTex3DTexel(handle, i1, j1, k0);
    const float t001 = pxgTex3DTexel(handle, i0, j0, k1);
    const float t101 = pxgTex3DTexel(handle, i1, j0, k1);
    const float t011 = pxgTex3DTexel(handle, i0, j1, k1);
    const float t111 = pxgTex3DTexel(handle, i1, j1, k1);

    const float a1 = 1.0f - a;
    const float b1 = 1.0f - b;
    const float g1 = 1.0f - g;

    // The guide's eight-term sum, term for term in its printed order, left to
    // right (K2: a convention).  The weights are exact; each product with a
    // texel and each add rounds once.
    float r = (a1 * b1 * g1) * t000;
    r = r + (a * b1 * g1) * t100;
    r = r + (a1 * b * g1) * t010;
    r = r + (a * b * g1) * t110;
    r = r + (a1 * b1 * g) * t001;
    r = r + (a * b1 * g) * t101;
    r = r + (a1 * b * g) * t011;
    r = r + (a * b * g) * t111;
    return r;
}

float pxgTex3D(uint handle, float x, float y, float z)
{
    return pxgTex3D(handle, float3(x, y, z));
}
