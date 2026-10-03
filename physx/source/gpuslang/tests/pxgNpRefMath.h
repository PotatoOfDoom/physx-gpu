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

// pxgNpRefMath.h -- the vector, quaternion and scalar math the NARROWPHASE host
// mirrors are built from, transcribed from the ENGINE and kept in one place.
//
// WHY THIS IS ADMISSIBLE AS AN ORACLE.  Every line comes from the PhysX foundation
// headers in this repository -- PxVec3.h, PxQuat.h, PxMath.h -- cited per function.
// A mirror of our own Slang would only prove the two copies agree with each other,
// which a differential between two copies of one mistake also proves.
//
// WHY IT IS A HEADER (P9.52b-5).  Five fixtures had grown a private copy of this
// block: testSphereHfCore, testHfTriFlagsBda, testSegTriangleBda, testCapTriangleBda
// and testCapsuleHfCore.  Before consolidating, all five copies of every shared
// function were extracted and diffed against each other expression by expression --
// float code is order-sensitive, so `a*b + c*d` against `c*d + a*b` would already be
// a divergence, as would a different association, a different comparison strictness
// or a guard written `!(m > 0)` instead of `m <= 0` (those differ on NaN).  NOT ONE
// arithmetic or control-flow divergence was found; every difference was formatting,
// an identifier name, or the extra trace out-parameters the `*Bda` probe fixtures
// carry.  The bodies below are the surviving copies moved VERBATIM, not retyped.
//
// The one thing consolidation gives up is redundancy: a mistake here now reaches
// five fixtures instead of one.  What replaces it is unchanged -- the mirror is
// written from the CUDA and the shader is written from the CUDA, separately, and
// each fixture still compares its shader against this mirror BITWISE.

#ifndef PXG_NP_REF_MATH_H
#define PXG_NP_REF_MATH_H

#include "foundation/PxQuat.h"

#include <cmath>
#include <cstdint>

namespace pxgtest
{
namespace npref
{

struct V3 { float x, y, z; };
struct V4 { float x, y, z, w; };

inline V3 sub(V3 a, V3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline V3 add(V3 a, V3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline V3 mul(V3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
inline V3 neg(V3 a) { return { -a.x, -a.y, -a.z }; }
inline float dot3(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross3(V3 a, V3 b)
{ return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }

// P9.94d: the RODRIGUES spelling these two used to carry, `v*(w*w - u.u) + u*(2 v.u)
// +- (u x v)*2w`.  It is not PxQuat's formula: the same rotation in exact arithmetic,
// a different float expression (last bits, and the sign of an exact zero).  Kept ONLY
// as the vacuity twin -- a fixture that flips p94dRodriguesTwin() and still matches
// its stored words has not observed the shader's switch to the literal formula.
inline V3 quatRotateRodrigues(V4 q, V3 v)
{
    const V3 u = { q.x, q.y, q.z };
    const float w = q.w;
    const float uu = dot3(u, u), vu = dot3(v, u);
    const V3 c = cross3(u, v);
    return { v.x * (w * w - uu) + u.x * (vu * 2.0f) + c.x * (w * 2.0f),
             v.y * (w * w - uu) + u.y * (vu * 2.0f) + c.y * (w * 2.0f),
             v.z * (w * w - uu) + u.z * (vu * 2.0f) + c.z * (w * 2.0f) };
}
inline V3 quatRotateInvRodrigues(V4 q, V3 v)
{
    const V3 u = { q.x, q.y, q.z };
    const float w = q.w;
    const float uu = dot3(u, u), vu = dot3(v, u);
    const V3 c = cross3(u, v);
    return { v.x * (w * w - uu) + u.x * (vu * 2.0f) - c.x * (w * 2.0f),
             v.y * (w * w - uu) + u.y * (vu * 2.0f) - c.y * (w * 2.0f),
             v.z * (w * w - uu) + u.z * (vu * 2.0f) - c.z * (w * 2.0f) };
}
// The twin switch.  False everywhere except inside a vacuity probe, which must set it
// back before returning.
inline bool& p94dRodriguesTwin() { static bool b = false; return b; }

// PxQuat::rotate / rotateInv (PxQuat.h:286-295 / :324-333) -- the engine header's
// own code, called rather than retyped; this TU is -ffp-contract=off, so the host
// evaluates it in exactly the order written.  P9.94d.
inline V3 quatRotate(V4 q, V3 v)
{
    if (p94dRodriguesTwin()) return quatRotateRodrigues(q, v);
    const physx::PxVec3 r = physx::PxQuat(q.x, q.y, q.z, q.w).rotate(physx::PxVec3(v.x, v.y, v.z));
    return { r.x, r.y, r.z };
}
inline V3 quatRotateInv(V4 q, V3 v)
{
    if (p94dRodriguesTwin()) return quatRotateInvRodrigues(q, v);
    const physx::PxVec3 r = physx::PxQuat(q.x, q.y, q.z, q.w).rotateInv(physx::PxVec3(v.x, v.y, v.z));
    return { r.x, r.y, r.z };
}
inline V4 quatMul(V4 a, V4 b)
{
    return { a.w * b.x + b.w * a.x + a.y * b.z - a.z * b.y,
             a.w * b.y + b.w * a.y + a.z * b.x - a.x * b.z,
             a.w * b.z + b.w * a.z + a.x * b.y - a.y * b.x,
             a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z };
}
inline V4 quatConj(V4 q) { return { -q.x, -q.y, -q.z, q.w }; }

// PxQuat::getBasisVector0 (PxQuat.h:260-265) -- ITS OWN FORMULA, not a rotate.
inline V3 basisVector0(V4 q)
{
    const float x2 = q.x * 2.0f;
    const float w2 = q.w * 2.0f;
    return V3{ (q.w * w2) - 1.0f + q.x * x2,
               (q.z * w2) + q.y * x2,
               (-q.y * w2) + q.z * x2 };
}

inline V3 normalized(V3 v)
{
    const float m = dot3(v, v);
    if (!(m > 0.0f)) return V3{ 0, 0, 0 };
    const float r = 1.0f / std::sqrt(m);
    return V3{ v.x * r, v.y * r, v.z * r };
}

inline float pxMax(float a, float b) { return a > b ? a : b; }   // PxMax<float> = intrinsics::selectMax
inline float pxMin(float a, float b) { return a < b ? a : b; }
inline float pxClamp(float v, float lo, float hi) { return pxMin(hi, pxMax(lo, v)); }

static const float MAX_F32 = 3.4028234663852886e+38f;

// validBarycentric (capsuleTriangle.cuh) -- the SAME epsilon on both bounds, and
// `v + w <= one` uses the INFLATED one, so a point just past the hypotenuse still
// counts as on the triangle.
// isValidTriangleBarycentricCoord (capsuleTriangle.cuh:102-110)
inline bool validBarycentric(float v, float w)
{
    const float eps = 1e-6f;
    const float zero = -eps;
    const float one = 1.0f + eps;
    return (v >= zero && v <= one) && (w >= zero && w <= one) && ((v + w) <= one);
}

}  // namespace npref
}  // namespace pxgtest

#endif  // PXG_NP_REF_MATH_H
