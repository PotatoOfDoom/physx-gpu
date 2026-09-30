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

#ifndef PXG_TESTS_REF_MATH_H
#define PXG_TESTS_REF_MATH_H

// CPU reference implementations of the math used by the ported kernels.
// These are direct C++ transcriptions of the CUDA device functions
// (preIntegration.cuh, integration.cuh, updateCacheAndBound.cuh and the
// PxQuat/PxTransform/PxMat33 foundation math) and serve as ground truth
// for the GPU verification tests.

#include "pxgGpuStructs.h"

#include <cmath>

namespace pxgtest
{

struct Vec3
{
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float v) : x(v), y(v), z(v) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3(const Float3& f) : x(f.x), y(f.y), z(f.z) {}

    Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator-() const { return Vec3(-x, -y, -z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const
    {
        return Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
    }
    Vec3 multiply(const Vec3& o) const { return Vec3(x * o.x, y * o.y, z * o.z); }
    float magnitudeSquared() const { return dot(*this); }
    float magnitude() const { return std::sqrt(magnitudeSquared()); }
    Vec3 getNormalized() const
    {
        const float m = magnitude();
        return m > 0.0f ? (*this) * (1.0f / m) : Vec3(0.0f);
    }
    Vec3 minimum(const Vec3& o) const
    {
        return Vec3(x < o.x ? x : o.x, y < o.y ? y : o.y, z < o.z ? z : o.z);
    }
    Vec3 maximum(const Vec3& o) const
    {
        return Vec3(x > o.x ? x : o.x, y > o.y ? y : o.y, z > o.z ? z : o.z);
    }
    Float3 toFloat3() const { return Float3{ x, y, z }; }
};

struct QuatV
{
    float x, y, z, w;

    QuatV() : x(0), y(0), z(0), w(1) {}
    QuatV(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    QuatV(const Quat& q) : x(q.x), y(q.y), z(q.z), w(q.w) {}
    QuatV(const Float4& q) : x(q.x), y(q.y), z(q.z), w(q.w) {}

    // mirrors PxQuat::operator* (Hamilton product)
    QuatV operator*(const QuatV& o) const
    {
        return QuatV(w * o.x + x * o.w + y * o.z - z * o.y,
                     w * o.y + y * o.w + z * o.x - x * o.z,
                     w * o.z + z * o.w + x * o.y - y * o.x,
                     w * o.w - x * o.x - y * o.y - z * o.z);
    }

    QuatV operator*(float s) const { return QuatV(x * s, y * s, z * s, w * s); }
    QuatV operator+(const QuatV& o) const { return QuatV(x + o.x, y + o.y, z + o.z, w + o.w); }

    QuatV getConjugate() const { return QuatV(-x, -y, -z, w); }

    QuatV getNormalized() const
    {
        const float s = 1.0f / std::sqrt(x * x + y * y + z * z + w * w);
        return QuatV(x * s, y * s, z * s, w * s);
    }

    // mirrors PxQuat::rotate
    Vec3 rotate(const Vec3& v) const
    {
        const float vx = 2.0f * v.x;
        const float vy = 2.0f * v.y;
        const float vz = 2.0f * v.z;
        const float w2 = w * w - 0.5f;
        const float dot2 = (x * vx + y * vy + z * vz);
        return Vec3((vx * w2 + (y * vz - z * vy) * w + x * dot2),
                    (vy * w2 + (z * vx - x * vz) * w + y * dot2),
                    (vz * w2 + (x * vy - y * vx) * w + z * dot2));
    }

    // mirrors PxQuat::rotateInv
    Vec3 rotateInv(const Vec3& v) const
    {
        const float vx = 2.0f * v.x;
        const float vy = 2.0f * v.y;
        const float vz = 2.0f * v.z;
        const float w2 = w * w - 0.5f;
        const float dot2 = (x * vx + y * vy + z * vz);
        return Vec3((vx * w2 - (y * vz - z * vy) * w + x * dot2),
                    (vy * w2 - (z * vx - x * vz) * w + y * dot2),
                    (vz * w2 - (x * vy - y * vx) * w + z * dot2));
    }

    // mirrors PxQuat::getBasisVector0
    Vec3 getBasisVector0() const
    {
        const float x2 = x * 2.0f;
        const float w2 = w * 2.0f;
        return Vec3((w * w2) - 1.0f + x * x2, (z * w2) + y * x2, (-y * w2) + z * x2);
    }

    Quat toQuat() const { return Quat{ x, y, z, w }; }
    Float4 toFloat4() const { return Float4{ x, y, z, w }; }
};

// column-major 3x3 (PxMat33)
struct Mat33
{
    Vec3 col0, col1, col2;

    Mat33() {}
    Mat33(const Vec3& c0, const Vec3& c1, const Vec3& c2) : col0(c0), col1(c1), col2(c2) {}

    // mirrors PxMat33(PxQuat)
    explicit Mat33(const QuatV& q)
    {
        const float x = q.x, y = q.y, z = q.z, w = q.w;
        const float x2 = x + x, y2 = y + y, z2 = z + z;
        const float xx = x2 * x, yy = y2 * y, zz = z2 * z;
        const float xy = x2 * y, xz = x2 * z, xw = x2 * w;
        const float yz = y2 * z, yw = y2 * w, zw = z2 * w;

        col0 = Vec3(1.0f - yy - zz, xy + zw, xz - yw);
        col1 = Vec3(xy - zw, 1.0f - xx - zz, yz + xw);
        col2 = Vec3(xz + yw, yz - xw, 1.0f - xx - yy);
    }

    Vec3 transform(const Vec3& v) const
    {
        return col0 * v.x + col1 * v.y + col2 * v.z;
    }

    Mat33 operator*(const Mat33& o) const
    {
        return Mat33(transform(o.col0), transform(o.col1), transform(o.col2));
    }

    Mat33 getTranspose() const
    {
        return Mat33(Vec3(col0.x, col1.x, col2.x), Vec3(col0.y, col1.y, col2.y), Vec3(col0.z, col1.z, col2.z));
    }

    float at(int row, int col) const
    {
        const Vec3& c = col == 0 ? col0 : (col == 1 ? col1 : col2);
        return row == 0 ? c.x : (row == 1 ? c.y : c.z);
    }

    void store(float out[9]) const
    {
        out[0] = col0.x; out[1] = col0.y; out[2] = col0.z;
        out[3] = col1.x; out[4] = col1.y; out[5] = col1.z;
        out[6] = col2.x; out[7] = col2.y; out[8] = col2.z;
    }

    static Mat33 load(const float in[9])
    {
        return Mat33(Vec3(in[0], in[1], in[2]), Vec3(in[3], in[4], in[5]), Vec3(in[6], in[7], in[8]));
    }
};

// PxTransform-like rigid transform
struct TransformV
{
    QuatV q;
    Vec3 p;

    TransformV() {}
    TransformV(const QuatV& q_, const Vec3& p_) : q(q_), p(p_) {}
    TransformV(const Transform28& t) : q(t.q), p(t.p) {}

    // mirrors PxTransform::transform(PxTransform) — no renormalization
    TransformV transform(const TransformV& x) const
    {
        return TransformV(q * x.q, q.rotate(x.p) + p);
    }

    // mirrors PxTransform::transformInv(PxTransform) LITERALLY (PxTransform.h:162-169):
    // the conjugate's rotate, not rotateInv -- the two differ in the sign of exact
    // zeros (P9.94a; pinned in testUpdateBounds' convexZeroTie test)
    TransformV transformInv(const TransformV& x) const
    {
        const QuatV qinv = q.getConjugate();
        return TransformV(qinv * x.q, qinv.rotate(x.p - p));
    }

    Transform28 toTransform28() const
    {
        Transform28 t;
        t.q = q.toQuat();
        t.p = p.toFloat3();
        return t;
    }
};

inline float fsel(float a, float b, float c) { return a >= 0.0f ? b : c; }

} // namespace pxgtest

#endif // PXG_TESTS_REF_MATH_H
