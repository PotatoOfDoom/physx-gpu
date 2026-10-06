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

// pxgTex3dRef.h -- P9.96, the host oracle for pxgTex3D (shaders/pxgSlangTex3d.slang).
//
// The SAME statement sequence as the shader, in fp32, compiled with the suite's
// -ffp-contract=off (tests/CMakeLists.txt) and with no std::fma and no promotion to
// double anywhere: the coordinate clamp to [0.5, N - 0.5] (K4), xB = xc - 0.5, floor,
// the 1.8 weight rounded to the nearest 1/256 with ties UP (K1), the address-safety
// index clamp, and the guide's eight-term sum left to right (K2).  See the module
// header for what is documented (the formula, the 9-bit format) and what is a choice
// (K0-K5).
//
// THE VARIANTS exist for bite asserts.  A fixture over this module must show, on its
// own data and before it launches anything, that every NON-EQUIVALENT variant gives a
// different value from V_GUIDE on at least one of its samples -- otherwise the
// matching shader mutant could survive and the fixture would not know (the P9.95d rule:
// wrong host spellings must change the expectation).  Each variant mirrors one row of
// the P9.96 mutant table.  Two spellings are EQUIVALENT and carry no bite assert:
//   * V_NAIVE_ROUND, floor(a*256 + 0.5)/256: equal to V_GUIDE for EVERY finite x,
//     brute-forced over all 4,278,190,080 finite floats through xB = x - 0.5, floor and
//     frac (it misrounds only a = 0x3affffff, which that path never produces);
//   * V_FLOOR_TRUNC, trunc(xB) for floor(xB): the coordinate clamp makes xB >= 0, where
//     the two agree (safety-only: it differs only for a NaN coordinate).
// V_IDX_HI_X (the x index clamp's upper bound missing) is NOT equivalent, although it
// moves only a weight-0 tap: the unclamped tap i + 1 = N is an out-of-bounds fetch,
// which llvmpipe returns as +0 (and this models), where the clamp reads T[N-1] -- and
// +0 * T[N-1] is -0 for a negative texel.  It changes the SIGN of a zero result at the
// far x face and nothing else, so only a fixture with such a -0 can bite on it.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace pxgtex
{

enum Variant : int
{
    V_GUIDE = 0,        // the shader
    V_TRUNC,            // weight truncated instead of rounded
    V_TIES_DOWN,        // r > 0.5 instead of r >= 0.5
    V_RNE,              // ties to even
    V_NOQUANT,          // no quantization (the old llvmpipe-like weights)
    V_Q9,               // 512 steps
    V_Q7,               // 128 steps
    V_NO_HALF,          // no -0.5 on any axis
    V_OFF_X, V_OFF_Y, V_OFF_Z,   // no -0.5 on one axis
    V_NESTED_G,         // nested lerps (1-a)x + a y, x then y then z
    V_NESTED_D,         // nested lerps x + a (y - x)
    V_SUM_ORDER,        // terms 2 and 3 of the sum swapped
    V_LAST_DROPPED,     // the abg T111 term missing
    V_B1_USES_A,        // (1 - a) where (1 - b) belongs
    V_TAP110_J,         // T[i+1, j, k] where T[i+1, j+1, k] belongs
    V_TAP011_K,         // T[i, j+1, k] where T[i, j+1, k+1] belongs
    V_Y_DIMX,           // the j1 index clamp against the x extent
    V_CC_GUIDE,         // coordinate clamp to the guide text's [0, N-1] on x
    V_CC_NONE,          // no coordinate clamp on any axis (index clamp only)
    V_CC_LO_X,          // x clamp: lower bound missing
    V_CC_HI_X,          // x clamp: upper bound missing
    V_CC_HI_N_Y,        // y clamp: upper bound N instead of N - 0.5
    V_CC_LO_0_Z,        // z clamp: lower bound 0 instead of 0.5
    V_IDX_HI_X,         // i1 index clamp missing (out-of-bounds fetch reads +0, as llvmpipe)
    V_NAIVE_ROUND,      // EQUIVALENT (see above)
    V_FLOOR_TRUNC,      // EQUIVALENT under the coordinate clamp (safety-only)
    V_COUNT
};

inline const char* variantName(int v)
{
    static const char* const n[V_COUNT] = {
        "guide", "w-trunc", "w-ties-down", "w-rne", "w-none", "w-bits9", "w-bits7", "no-half",
        "off-x-missing", "off-y-missing", "off-z-missing", "lerp-nested-guide", "lerp-nested-diff",
        "sum-order-swap", "last-term-dropped", "w-b1-uses-a", "tap-110-wrong-j", "tap-011-wrong-k",
        "clamp-y-uses-dimx", "cc-guide", "cc-none", "cc-lo-x-missing", "cc-hi-x-missing", "cc-hi-is-N",
        "cc-lo-is-0", "idx-clamp-hi-missing", "w-naive-EQUIV", "floor-x-trunc-SAFETY" };
    return (v >= 0 && v < V_COUNT) ? n[v] : "?";
}
inline bool variantIsEquivalent(int v) { return v == V_GUIDE || v == V_NAIVE_ROUND || v == V_FLOOR_TRUNC; }
// Whether the variant clamps all three COORDINATES.  The three that do not convert an
// unclamped coordinate to an index, which is undefined (in C++ as on the GPU) for an
// infinite or huge one, so a fixture with such coordinates skips them for these.
inline bool variantClampsAllCoords(int v) { return v != V_CC_NONE && v != V_CC_LO_X && v != V_CC_HI_X; }

struct Vol                      // texel values AFTER the format read
{
    uint32_t w = 0, h = 0, d = 0;
    std::vector<float> t;       // x fastest
    float at(int x, int y, int z) const { return t[(size_t(z) * h + size_t(y)) * w + size_t(x)]; }
};

// The UNORM read: the engine's own spelling (Gu::SDF::decodeSample, GuSDF.h:167/:171),
// which is also what Mesa's lp_build_unsigned_norm_to_float computes, and which
// npTex3dBindlessUnormExhaustive MEASURED llvmpipe's Load to return for every code
// (outcome A of the P9.96 plan).  NOT the correctly rounded quotient v/255.
inline float unorm8(uint32_t v) { return float(v) * (1.0f / 255.0f); }
inline float unorm16(uint32_t v) { return float(v) * (1.0f / 65535.0f); }

inline float quantize(float a, int v)
{
    if (v == V_NOQUANT) return a;
    const float S = (v == V_Q9) ? 512.0f : (v == V_Q7) ? 128.0f : 256.0f;
    const float step = 1.0f / S;   // exact: S is a power of two
    if (v == V_NAIVE_ROUND) return std::floor(a * S + 0.5f) * step;
    const float s = a * S;
    const float q = std::floor(s);
    const float r = s - q;
    float up;
    switch (v)
    {
    case V_TRUNC:     up = 0.0f; break;
    case V_TIES_DOWN: up = (r > 0.5f) ? 1.0f : 0.0f; break;
    case V_RNE:       up = (r > 0.5f || (r == 0.5f && std::fmod(q, 2.0f) != 0.0f)) ? 1.0f : 0.0f; break;
    default:          up = (r >= 0.5f) ? 1.0f : 0.0f; break;    // K1: nearest, ties UP
    }
    return (q + up) * step;
}

inline int clampIdx(int i, uint32_t n) { return std::min(std::max(i, 0), std::max(int(n) - 1, 0)); }

// K4, per axis.  lo/hi are the variant's bounds; a missing bound is +-inf.
inline float clampCoord(float x, float lo, float hi) { return std::min(std::max(x, lo), hi); }

inline float lerpG(float a, float x, float y) { return (1.0f - a) * x + a * y; }
inline float lerpD(float a, float x, float y) { return x + a * (y - x); }

// CUDA tex3D<float>, unnormalized coordinates, clamp, linear -- the Programming Guide's
// "Texture Fetching / Linear Filtering" read in IEEE fp32.  NON-NaN coordinates: an
// infinite or huge one is clamped like any other and gives the edge sample, as the
// shader's FMin/FMax do (pxgSlangTex3d K5).  Not a NaN coordinate (its float-to-int
// conversion is undefined in C++, and the GPU's value is unspecified; the GPU lanes
// that test it are checked for "written" only), and for the variants without a full
// coordinate clamp (variantClampsAllCoords) only |coordinate| < 2^31.
inline float tex3D(const Vol& V, float x, float y, float z, int v = V_GUIDE)
{
    const float INF = INFINITY;
    const float nx = float(V.w), ny = float(V.h), nz = float(V.d);
    float xlo = 0.5f, xhi = nx - 0.5f, ylo = 0.5f, yhi = ny - 0.5f, zlo = 0.5f, zhi = nz - 0.5f;
    switch (v)
    {
    case V_CC_GUIDE:  xlo = 0.0f; xhi = nx - 1.0f; break;
    case V_CC_NONE:   xlo = ylo = zlo = -INF; xhi = yhi = zhi = INF; break;
    case V_CC_LO_X:   xlo = -INF; break;
    case V_CC_HI_X:   xhi = INF; break;
    case V_CC_HI_N_Y: yhi = ny; break;
    case V_CC_LO_0_Z: zlo = 0.0f; break;
    default: break;
    }
    const float xc = clampCoord(x, xlo, xhi);
    const float yc = clampCoord(y, ylo, yhi);
    const float zc = clampCoord(z, zlo, zhi);

    const bool noHalf = v == V_NO_HALF;
    const float xB = (noHalf || v == V_OFF_X) ? xc : xc - 0.5f;
    const float yB = (noHalf || v == V_OFF_Y) ? yc : yc - 0.5f;
    const float zB = (noHalf || v == V_OFF_Z) ? zc : zc - 0.5f;
    const float fx = (v == V_FLOOR_TRUNC) ? std::trunc(xB) : std::floor(xB);
    const float fy = std::floor(yB);
    const float fz = std::floor(zB);
    const float a = quantize(xB - fx, v);
    const float b = quantize(yB - fy, v);
    const float g = quantize(zB - fz, v);

    const int i0 = clampIdx(int(fx), V.w);
    const int i1 = (v == V_IDX_HI_X) ? int(fx) + 1 : clampIdx(int(fx) + 1, V.w);
    const int j0 = clampIdx(int(fy), V.h);
    const int j1 = clampIdx(int(fy) + 1, v == V_Y_DIMX ? V.w : V.h);
    const int k0 = clampIdx(int(fz), V.d), k1 = clampIdx(int(fz) + 1, V.d);
    // V_Y_DIMX may name a row, and V_IDX_HI_X a column, past the volume; the GPU reads
    // such a texel as +0 (llvmpipe returns zero for an out-of-bounds fetch), and so does
    // this.  No other variant leaves the volume.
    auto T = [&](int i, int j, int k) -> float {
        return (i >= int(V.w) || j >= int(V.h)) ? 0.0f : V.at(i, j, k);
    };

    const float t000 = T(i0, j0, k0), t100 = T(i1, j0, k0), t010 = T(i0, j1, k0);
    const float t110 = (v == V_TAP110_J) ? T(i1, j0, k0) : T(i1, j1, k0);
    const float t001 = T(i0, j0, k1), t101 = T(i1, j0, k1);
    const float t011 = (v == V_TAP011_K) ? T(i0, j1, k0) : T(i0, j1, k1);
    const float t111 = T(i1, j1, k1);

    if (v == V_NESTED_G || v == V_NESTED_D)
    {
        auto L = (v == V_NESTED_G) ? lerpG : lerpD;
        const float x00 = L(a, t000, t100), x10 = L(a, t010, t110), x01 = L(a, t001, t101), x11 = L(a, t011, t111);
        return L(g, L(b, x00, x10), L(b, x01, x11));
    }
    const float a1 = 1.0f - a;
    const float b1 = (v == V_B1_USES_A) ? 1.0f - a : 1.0f - b;
    const float g1 = 1.0f - g;
    float r = (a1 * b1 * g1) * t000;
    if (v == V_SUM_ORDER)
    {
        r = r + (a1 * b * g1) * t010;
        r = r + (a * b1 * g1) * t100;
    }
    else
    {
        r = r + (a * b1 * g1) * t100;
        r = r + (a1 * b * g1) * t010;
    }
    r = r + (a * b * g1) * t110;
    r = r + (a1 * b1 * g) * t001;
    r = r + (a * b1 * g) * t101;
    r = r + (a1 * b * g) * t011;
    if (v != V_LAST_DROPPED)
        r = r + (a * b * g) * t111;
    return r;
}

} // namespace pxgtex
