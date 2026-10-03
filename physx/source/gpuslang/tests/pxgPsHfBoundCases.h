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

// pxgPsHfBoundCases.h -- P9.94c: the directed particle-vs-heightfield inputs that make
// the local-bound spelling of particleSystemHFMidPhaseCG.cu:109-122 observable.
//
//     localBound = PxBounds3::transformFast(heightfieldTransform.getInverse(), worldBound);
//
// getInverse() is (q.rotateInv(-p), conj q), and transformFast is
// basisExtent(inv.q.rotate(c) + inv.p, PxMat33(inv.q), extents).  The pre-P9.94c port
// rotated (c - p) once and took the basis from rotateInv(q, e_i).  Under a GENERAL pose
// every cell an ULP of the bound can toggle lies further than the contact distance
// from the particle, so no contact moves; only AXIS-ALIGNED poses (where each row of
// |R| has L1 norm 1) put a contact on the knife edge.  Each pose below has two
// particles, both on the integer row boundary 3 in heightfield space, whose one-way
// contact COUNT depends on the spelling.
//
// Shared scene (all poses): 24 x 17 grid, row heights 30, 20, 10 for rows 0-2 and 0
// after, materials 0, tessellation flag 0, scale (1, 1, 1), restDist 0.04f, no CCD,
// predicted == current.  Shape-space y is exactly 0 for every particle, so the flat
// cells see ppd == +-0 (no contact) and only the sloped row-2 cell can contact.
//
// Counts per spelling (the five HfBoundSpelling values in testPsHfCollide.cpp):
//   nLit     -- the CUDA (literal) spelling,
//   nCentre  -- centre reverted to rotateInv(q, c - p), basis literal,
//   nBasis   -- basis reverted to rotateInv(q, e_i), centre literal,
//   nOld     -- both reverted (the pre-P9.94c port),
//   nNoConj  -- the literal chain with PxMat33(q) instead of PxMat33(conj q).
// X90 P1 kills the basis revert, DIAG A/B the centre and full reverts, DIAG035 N1 the
// dropped conjugate (a single-axis turn cannot see it: R and R^T differ only in the
// signs abs() removes).  The X90 quaternion is (s, 0, 0, s) with s = 0x3f3504f3, whose
// norm^2 is 0.99999994 -- the PxMat33 diagonal is exactly 1 and the rotateInv one is
// not, and that is what X90 P1 sees.  Nothing on the way normalises it.
//
// Raw bit patterns only: both including files have their own float vector types.

#pragma once

#include <cstdint>

namespace pxgpshfbound
{

struct Part
{
    uint32_t cur[3];                                  // world position (w = 1)
    uint32_t nLit, nCentre, nBasis, nOld, nNoConj;    // one-way contact count per spelling
};

struct Pose
{
    const char* name;
    uint32_t q[4];          // heightfield rotation (x, y, z, w)
    uint32_t p[3];          // heightfield position
    uint32_t cdPs, cdHf;    // the two halves of contactDist, summed on the host in float
    Part part[2];
};

static constexpr uint32_t kRows = 24u, kCols = 17u;
static constexpr int16_t kRowHeight[3] = { 30, 20, 10 };   // rows >= 3 are 0
static constexpr float kRestDist = 0.04f;

inline constexpr Pose kPoses[3] = {
    { "X90",
      { 0x3f3504f3u, 0x00000000u, 0x00000000u, 0x3f3504f3u },
      { 0xc0df9415u, 0xc0d03f8au, 0x409dc735u },
      0x3f000000u, 0x3f000000u,   // 0.5f + 0.5f
      { { { 0xc03f2829u, 0xc0d03f8au, 0x409dc735u }, 2u, 2u, 0u, 2u, 2u },     // P1
        { { 0xc03f2828u, 0xc0d03f8au, 0x409dc735u }, 0u, 2u, 0u, 0u, 0u } } }, // P2
    { "DIAG",
      { 0x3f000000u, 0x3f000000u, 0x3f000000u, 0x3f000000u },
      { 0x41f39d4bu, 0xc2e26882u, 0x42c2ca89u },
      0x3e800000u, 0x3e800000u,   // 0.25f + 0.25f
      { { { 0x41ff9d4bu, 0xc2db6882u, 0x42c2ca89u }, 1u, 0u, 1u, 0u, 1u },     // A
        { { 0x41f79d4bu, 0xc2db6883u, 0x42c2ca89u }, 0u, 1u, 0u, 1u, 0u } } }, // B
    { "DIAG035",
      { 0x3f000000u, 0x3f000000u, 0x3f000000u, 0x3f000000u },
      { 0x41f38325u, 0xc2f6a488u, 0x41bd9b40u },
      0x3e570a3du, 0x3e0f5c29u,   // 0.21f + 0.14f (= 0x3eb33333)
      { { { 0x41f38325u, 0xc2eff155u, 0x41bd9b40u }, 0u, 0u, 0u, 0u, 2u },     // N1
        { { 0x41f38325u, 0xc2eff155u, 0x41bd9b41u }, 3u, 1u, 3u, 1u, 3u } } }, // L1
};

} // namespace pxgpshfbound
