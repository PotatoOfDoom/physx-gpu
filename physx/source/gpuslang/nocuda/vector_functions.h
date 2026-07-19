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

// vector_functions.h — RUNTIME_PLAN.md P4: no-CUDA-toolkit stand-in for the CUDA
// toolkit header of the same name.  Supplies the make_<type>(...) component
// constructors (make_float4, make_uint3, ...) that gpu* HOST helpers such as
// cutil_math.h use to build the POD vector types from vector_types.h.  The
// signatures and semantics match the toolkit header exactly (each simply packs
// its arguments into the struct's x/y/z/w members in order).  On the include path
// only for the no-toolkit targets; the real toolkit header wins when present.

#ifndef PXG_NOCUDA_VECTOR_FUNCTIONS_H
#define PXG_NOCUDA_VECTOR_FUNCTIONS_H

#include "host_defines.h"
#include "vector_types.h"

// One constructor per component count, matching cuda.h's make_<name><N> family.
#define PXG_MAKE_VEC(base, name)                                                                  \
    static inline name##1 make_##name##1(base x) { name##1 t; t.x = x; return t; }                \
    static inline name##2 make_##name##2(base x, base y) { name##2 t; t.x = x; t.y = y; return t; } \
    static inline name##3 make_##name##3(base x, base y, base z)                                   \
    { name##3 t; t.x = x; t.y = y; t.z = z; return t; }                                            \
    static inline name##4 make_##name##4(base x, base y, base z, base w)                           \
    { name##4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t; }

PXG_MAKE_VEC(signed char, char)
PXG_MAKE_VEC(unsigned char, uchar)
PXG_MAKE_VEC(short, short)
PXG_MAKE_VEC(unsigned short, ushort)
PXG_MAKE_VEC(int, int)
PXG_MAKE_VEC(unsigned int, uint)
PXG_MAKE_VEC(long int, long)
PXG_MAKE_VEC(unsigned long int, ulong)
PXG_MAKE_VEC(long long int, longlong)
PXG_MAKE_VEC(unsigned long long int, ulonglong)
PXG_MAKE_VEC(float, float)
PXG_MAKE_VEC(double, double)

#undef PXG_MAKE_VEC

#endif // PXG_NOCUDA_VECTOR_FUNCTIONS_H
