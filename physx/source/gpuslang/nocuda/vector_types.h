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

// vector_types.h — RUNTIME_PLAN.md P4: no-CUDA-toolkit stand-in for the CUDA
// toolkit header of the same name.  The gpu* HOST modules (e.g. PxgIntegerAABB.h)
// use CUDA's POD vector types (uint4, float4, ...) for buffer layout.  In the
// Vulkan / SPIR-V build there is no CUDA toolkit on the include path, so this
// dir (added to the include path only for the no-toolkit targets) supplies a
// faithful reproduction of the toolkit's public struct ABI: identical member
// order and identical alignment (float4/int4/uint4 16-byte aligned, the 2-vectors
// 8-byte aligned, the 3-vectors naturally aligned), so host-filled buffers keep
// the exact byte layout the ported Slang kernels read.  Only host-side POD types
// and dim3 are provided; the toolkit's device intrinsics are never referenced by
// host code.  When the real toolkit IS present its header wins (this dir is not
// on the include path), so the CUDA build is unaffected.

#ifndef PXG_NOCUDA_VECTOR_TYPES_H
#define PXG_NOCUDA_VECTOR_TYPES_H

#if defined(__GNUC__) || defined(__clang__)
#define PXG_VT_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define PXG_VT_ALIGN(n) __declspec(align(n))
#else
#define PXG_VT_ALIGN(n)
#endif

// --- 1/2/3/4-component vector types (member order + alignment match cuda.h) ---

struct char1 { signed char x; };
struct uchar1 { unsigned char x; };
struct PXG_VT_ALIGN(2) char2 { signed char x, y; };
struct PXG_VT_ALIGN(2) uchar2 { unsigned char x, y; };
struct char3 { signed char x, y, z; };
struct uchar3 { unsigned char x, y, z; };
struct PXG_VT_ALIGN(4) char4 { signed char x, y, z, w; };
struct PXG_VT_ALIGN(4) uchar4 { unsigned char x, y, z, w; };

struct short1 { short x; };
struct ushort1 { unsigned short x; };
struct PXG_VT_ALIGN(4) short2 { short x, y; };
struct PXG_VT_ALIGN(4) ushort2 { unsigned short x, y; };
struct short3 { short x, y, z; };
struct ushort3 { unsigned short x, y, z; };
struct PXG_VT_ALIGN(8) short4 { short x, y, z, w; };
struct PXG_VT_ALIGN(8) ushort4 { unsigned short x, y, z, w; };

struct int1 { int x; };
struct uint1 { unsigned int x; };
struct PXG_VT_ALIGN(8) int2 { int x, y; };
struct PXG_VT_ALIGN(8) uint2 { unsigned int x, y; };
struct int3 { int x, y, z; };
struct uint3 { unsigned int x, y, z; };
struct PXG_VT_ALIGN(16) int4 { int x, y, z, w; };
struct PXG_VT_ALIGN(16) uint4 { unsigned int x, y, z, w; };

struct long1 { long int x; };
struct ulong1 { unsigned long int x; };
struct PXG_VT_ALIGN(2 * sizeof(long int)) long2 { long int x, y; };
struct PXG_VT_ALIGN(2 * sizeof(unsigned long int)) ulong2 { unsigned long int x, y; };
struct long3 { long int x, y, z; };
struct ulong3 { unsigned long int x, y, z; };
struct PXG_VT_ALIGN(16) long4 { long int x, y, z, w; };
struct PXG_VT_ALIGN(16) ulong4 { unsigned long int x, y, z, w; };

struct longlong1 { long long int x; };
struct ulonglong1 { unsigned long long int x; };
struct PXG_VT_ALIGN(16) longlong2 { long long int x, y; };
struct PXG_VT_ALIGN(16) ulonglong2 { unsigned long long int x, y; };
struct longlong3 { long long int x, y, z; };
struct ulonglong3 { unsigned long long int x, y, z; };
struct PXG_VT_ALIGN(16) longlong4 { long long int x, y, z, w; };
struct PXG_VT_ALIGN(16) ulonglong4 { unsigned long long int x, y, z, w; };

struct float1 { float x; };
struct PXG_VT_ALIGN(8) float2 { float x, y; };
struct float3 { float x, y, z; };
struct PXG_VT_ALIGN(16) float4 { float x, y, z, w; };

struct double1 { double x; };
struct PXG_VT_ALIGN(16) double2 { double x, y; };
struct double3 { double x, y, z; };
struct PXG_VT_ALIGN(16) double4 { double x, y, z, w; };

// --- dim3 (grid/block launch dims), matches cuda.h's host definition ---

struct dim3
{
    unsigned int x, y, z;
#if defined(__cplusplus)
    dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    operator uint3(void) const { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif
};

#undef PXG_VT_ALIGN

#endif // PXG_NOCUDA_VECTOR_TYPES_H
