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

// host_defines.h — RUNTIME_PLAN.md P4: no-CUDA-toolkit stand-in for the CUDA
// toolkit header of the same name.  A handful of gpu* HOST headers (notably the
// Cg-style cutil_math.h vector helpers) decorate their inline functions with the
// raw CUDA execution-space qualifiers __device__ / __host__.  In the real build
// those .cpp translation units are compiled by nvcc, which supplies these macros;
// in the Vulkan / no-toolkit build the host compiler sees them, so this stand-in
// (on the include path only for the no-toolkit targets) expands them to nothing —
// exactly what the toolkit's own host_defines.h does when __CUDACC__ is undefined.
// Only the qualifiers that are NOT native compiler keywords are defined here; the
// gcc/clang keywords __inline__ / __restrict__ are deliberately left untouched --
// except under MSVC, which has neither (cutil_math.h uses `static __inline__`),
// where they map to its own spellings exactly as the toolkit's header does.

#ifndef PXG_NOCUDA_HOST_DEFINES_H
#define PXG_NOCUDA_HOST_DEFINES_H

#ifndef __location__
#define __location__(...)
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#ifndef __inline__
#define __inline__ __inline
#endif
#ifndef __restrict__
#define __restrict__ __restrict
#endif
#endif

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __constant__
#define __constant__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __managed__
#define __managed__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif
#ifndef __noinline__
#define __noinline__
#endif
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
#ifndef __align__
#if defined(__GNUC__) || defined(__clang__)
#define __align__(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define __align__(n) alignas(n) // not __declspec(align(n)): that takes integer literals only
#else
#define __align__(n)
#endif
#endif
// nvcc exposes the alignment attribute a second time under this spelling; some
// engine structs (e.g. PxgMaterialContactData) tag their layout with it.
#ifndef __builtin_align__
#define __builtin_align__(n) __align__(n)
#endif

#endif // PXG_NOCUDA_HOST_DEFINES_H
