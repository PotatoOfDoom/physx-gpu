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

// pxgHfLocalBoundRef.h -- P9.89: the deformable-vs-heightfield midphase's world-to-
// heightfield bound, with the ENGINE's own foundation code as the oracle, plus twins of
// the port's pre-P9.89 operation order and a knife-edge search that makes them visible.
//
// THE UPSTREAM CHAIN (femClothHFMidPhase.cu:107-129, softbodyHFMidPhase.cu:205-227):
//
//     localBound = PxBounds3::transformFast(heightfieldTransform.getInverse(), bound);
//     localBound.{minimum,maximum}.{x,y,z} *= 1/|sx|, 1/sy, 1/|sz|;  swap on negative scale
//
// getInverse() is (q.rotateInv(-p), conj q) and transformFast is
// basisExtent(inv.q.rotate(c) + inv.p, PxMat33(inv.q), extents).  The CUDA therefore
// rotates the centre and -p SEPARATELY and adds the two rounded vectors, and builds the
// basis with PxMat33(PxQuat)'s `(1 - yy) - zz` diagonal.  Until P9.89 the port rotated
// (c - p) once and took the basis columns from rotateInv(q, e_i), whose diagonal is the
// `2(w^2 - 0.5) + 2x^2` identity instead.  Equal in exact arithmetic, a few ULPs apart in
// float -- and an ULP at the wrong place is a floor()/ceil() on the other side of an
// integer, i.e. a different pair SET.
//
// The oracle below does not transcribe transformFast: it CALLS it, from the same
// foundation headers the CUDA compiles against, in a binary built with
// -ffp-contract=off.  Only the twins are spelled out by hand.
//
// WHY A SEARCH.  A generic rotation moves ~95% of all bounds by an ULP or two, but a pair
// set only changes when an integer row/column/height falls between the two results, which
// a random point does about once in a million.  `findKnife` places a point so that one
// chosen bound lands on an integer and then walks the world point ULP by ULP until the
// CUDA and the requested twin emit different pairs.  The fixtures ASSERT that it found
// one for every twin, so a fixture that lost its teeth fails instead of passing.

#ifndef PXG_HF_LOCAL_BOUND_REF_H
#define PXG_HF_LOCAL_BOUND_REF_H

#include "foundation/PxBounds3.h"
#include "foundation/PxMat33.h"
#include "foundation/PxQuat.h"
#include "foundation/PxTransform.h"
#include "foundation/PxVec3.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace pxgtest
{
namespace hflb
{

// Which pieces of the PRE-P9.89 port order to substitute.  0 is the CUDA.
enum : uint32_t
{
    CUDA       = 0u,
    OLD_CENTRE = 1u,   // rotateInv(q, c - p)       instead of conj(q).rotate(c) + rotateInv(q, -p)
    OLD_COL0   = 2u,   // rotateInv(q, (1,0,0))     instead of PxMat33(conj q).column0
    OLD_COL1   = 4u,   // rotateInv(q, (0,1,0))     instead of PxMat33(conj q).column1
    OLD_COL2   = 8u,   // rotateInv(q, (0,0,1))     instead of PxMat33(conj q).column2
    OLD_BASIS  = OLD_COL0 | OLD_COL1 | OLD_COL2,
    OLD_ALL    = OLD_CENTRE | OLD_BASIS,   // the port before P9.89
    // The hand-composed chain with NOTHING substituted.  It must equal the foundation's
    // transformFast bit for bit -- the fixtures assert that, which is what licenses reading
    // the OLD_* twins as "the CUDA with exactly one piece changed".
    COMPOSED   = 16u
};

struct Placement
{
    physx::PxQuat q;
    physx::PxVec3 p;
    physx::PxVec3 scale;   // row, height, column
};

// The heightfield blob: nbRows @0, nbCols @4, then 4-byte PxHeightFieldSample words.
struct Field
{
    const uint32_t* blob;
    uint32_t nbRows() const { return blob[0]; }
    uint32_t nbCols() const { return blob[1]; }
    float height(uint32_t vi) const { return float(int16_t(uint16_t(blob[2u + vi] & 0xffffu))); }
    uint32_t material(uint32_t vi, bool slot1) const
    {
        return (blob[2u + vi] >> (slot1 ? 24 : 16)) & 0x7fu;   // PxBitAndByte: tess flag off
    }
};

// cvt.rzi.u32.f32 -- the implicit PxReal -> PxU32 of heightfieldComputePairs' miny/maxy.
inline uint32_t f2uSat(float x)
{
    if (!(x > 0.0f)) return 0u;
    if (x >= 4294967296.0f) return 0xffffffffu;
    return uint32_t(x);
}

// femClothHFMidPhase.cu:101-129 (softbodyHFMidPhase.cu:199-227 is character for character
// the same).  With variant == CUDA the transform is the foundation's own; any other value
// goes through the hand-composed chain with the OLD_* pieces it names swapped in.
inline void localBound(const Placement& hf, const physx::PxBounds3& wb, uint32_t variant,
                       physx::PxVec3& lMin, physx::PxVec3& lMax)
{
    using namespace physx;
    const PxReal oneOverHeightScale = 1.f / hf.scale.y;
    const PxReal oneOverRowScale = 1.f / PxAbs(hf.scale.x);
    const PxReal oneOverColScale = 1.f / PxAbs(hf.scale.z);

    const PxTransform hfTransform(hf.p, hf.q);
    PxBounds3 lb;
    if (variant == CUDA)
    {
        lb = PxBounds3::transformFast(hfTransform.getInverse(), wb);
    }
    else
    {
        const PxTransform inv = hfTransform.getInverse();
        const PxVec3 c = wb.getCenter(), e = wb.getExtents();
        const PxVec3 lc = (variant & OLD_CENTRE) ? hf.q.rotateInv(c - hf.p) : inv.transform(c);
        const PxMat33 m(inv.q);
        const PxVec3 b0 = (variant & OLD_COL0) ? hf.q.rotateInv(PxVec3(1.f, 0.f, 0.f)) : m.column0;
        const PxVec3 b1 = (variant & OLD_COL1) ? hf.q.rotateInv(PxVec3(0.f, 1.f, 0.f)) : m.column1;
        const PxVec3 b2 = (variant & OLD_COL2) ? hf.q.rotateInv(PxVec3(0.f, 0.f, 1.f)) : m.column2;
        lb = PxBounds3::basisExtent(lc, PxMat33(b0, b1, b2), e);
    }
    lb.minimum.x *= oneOverRowScale;
    lb.minimum.y *= oneOverHeightScale;
    lb.minimum.z *= oneOverColScale;
    lb.maximum.x *= oneOverRowScale;
    lb.maximum.y *= oneOverHeightScale;
    lb.maximum.z *= oneOverColScale;
    if (hf.scale.x < 0.f) { const PxReal t = lb.minimum.x; lb.minimum.x = lb.maximum.x; lb.maximum.x = t; }
    if (hf.scale.z < 0.f) { const PxReal t = lb.minimum.z; lb.minimum.z = lb.maximum.z; lb.maximum.z = t; }
    lMin = lb.minimum;
    lMax = lb.maximum;
}

// The admit gate (with upstream's bracketing), getMin/MaxRow/Column (heightfieldUtil.cuh:37-58)
// and heightfieldOutputPairs (heightfieldUtil.cuh:382) -- the triangle indices one element
// emits, in emission order.
inline void pairs(const Field& f, const physx::PxVec3& mn, const physx::PxVec3& mx,
                  std::vector<uint32_t>& out)
{
    const uint32_t nbRows = f.nbRows(), nbCols = f.nbCols();
    if (!(!(mn.x > float(nbRows - 1u)) || (mn.z > float(nbCols - 1u)) || (mx.x < 0.f) || (mx.z < 0.f)))
        return;
    auto clampI = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    const uint32_t minRow = uint32_t(clampI(int(std::floor(mn.x)), 0, int(nbRows) - 2));
    const uint32_t maxRow = uint32_t(clampI(int(std::ceil(mx.x)), 0, int(nbRows) - 1));
    const uint32_t minCol = uint32_t(clampI(int(std::floor(mn.z)), 0, int(nbCols) - 2));
    const uint32_t maxCol = uint32_t(clampI(int(std::ceil(mx.z)), 0, int(nbCols) - 1));
    if (!((2u * (maxCol - minCol) * (maxRow - minRow)) > 0u)) return;
    const float miny = float(f2uSat(mn.y)), maxy = float(f2uSat(mx.y));
    const uint32_t columnSpan = maxCol - minCol;
    const uint32_t total = (maxRow - minRow) * columnSpan * 2u;
    for (uint32_t i = 0; i < total; ++i)
    {
        const uint32_t index = i / 2u;
        const uint32_t vi = (minRow + index / columnSpan) * nbCols + (minCol + index % columnSpan);
        const float h0 = f.height(vi), h1 = f.height(vi + 1u);
        const float h2 = f.height(vi + nbCols), h3 = f.height(vi + nbCols + 1u);
        const bool con0 = maxy < h0 && maxy < h1 && maxy < h2 && maxy < h3;
        const bool con1 = miny > h0 && miny > h1 && miny > h2 && miny > h3;
        if (con0 || con1) continue;
        const bool slot1 = (i & 1u) != 0u;
        if (f.material(vi, slot1) == 127u) continue;    // PxHeightFieldMaterial::eHOLE
        out.push_back((vi << 1) + (slot1 ? 1u : 0u));
    }
}

// The element bound both fixtures feed: a sphere of `radius` around one world point.  The
// cloth VERTEX path builds it as (p - r, p + r) (femClothHFMidPhase.cu:379-380); the softbody
// path gets the same bits from a tetrahedron whose four indices name one vertex, through
// tetBoundingBox + fattenFast (softbodyHFMidPhase.cu:394-396).
inline physx::PxBounds3 sphereBound(const physx::PxVec3& p, float radius)
{
    const physx::PxVec3 r(radius);
    return physx::PxBounds3(p - r, p + r);
}

inline void pairsOfPoint(const Placement& hf, const Field& f, const physx::PxVec3& p,
                         float radius, uint32_t variant, std::vector<uint32_t>& out)
{
    physx::PxVec3 mn, mx;
    localBound(hf, sphereBound(p, radius), variant, mn, mx);
    pairs(f, mn, mx, out);
}

inline bool separates(const Placement& hf, const Field& f, const physx::PxVec3& p, float radius,
                      uint32_t variant)
{
    std::vector<uint32_t> a, b;
    pairsOfPoint(hf, f, p, radius, CUDA, a);
    pairsOfPoint(hf, f, p, radius, variant, b);
    return a != b;
}

inline float stepUlps(float x, int n)
{
    for (; n > 0; --n) x = std::nextafter(x, 3.0e38f);
    for (; n < 0; ++n) x = std::nextafter(x, -3.0e38f);
    return x;
}

// Heightfield units (row, height, column) -> world, forward transform.  Positive scales only;
// the search is not meant for mirrored fields.
inline physx::PxVec3 toWorld(const Placement& hf, const physx::PxVec3& hfUnits)
{
    const physx::PxVec3 s(hfUnits.x * hf.scale.x, hfUnits.y * hf.scale.y, hfUnits.z * hf.scale.z);
    return hf.q.rotate(s) + hf.p;
}

// Find a world point near `aroundHf` (heightfield units) whose CUDA bound puts the chosen
// face -- `axis` 0 row, 1 height, 2 column; `useMax` min or max -- on the integer `k`, and
// whose pair set under `variant` differs from the CUDA's.  Deterministic: IEEE float, no
// contraction, a fixed scan order.
inline bool findKnife(const Placement& hf, const Field& f, float radius, uint32_t axis,
                      bool useMax, float k, physx::PxVec3 aroundHf, uint32_t variant,
                      physx::PxVec3& outP, int span = 7)
{
    using namespace physx;
    for (int it = 0; it < 4; ++it)
    {
        PxVec3 mn, mx;
        localBound(hf, sphereBound(toWorld(hf, aroundHf), radius), CUDA, mn, mx);
        const float v = useMax ? mx[axis] : mn[axis];
        aroundHf[axis] += (k - v);
    }
    const PxVec3 p0 = toWorld(hf, aroundHf);
    for (int dx = -span; dx <= span; ++dx)
        for (int dy = -span; dy <= span; ++dy)
            for (int dz = -span; dz <= span; ++dz)
            {
                const PxVec3 p(stepUlps(p0.x, dx), stepUlps(p0.y, dy), stepUlps(p0.z, dz));
                if (separates(hf, f, p, radius, variant))
                {
                    outP = p;
                    return true;
                }
            }
    return false;
}

// ---------------------------------------------------------------------------
// The scene both *HeightfieldMidphaseRotated fixtures drive.
//
// THE ROTATION IS CHOSEN, NOT GENERIC.  About 0.35 rad (a 0.7 rad turn) about a skew
// axis, and picked among its neighbours because all THREE diagonal entries of
// PxMat33(conj q) differ from the rotateInv(q, e_i) spelling -- by two ULPs each.  The
// first candidate tried had all three diagonals bitwise equal, so the basis half of the
// old port could not have shown there whatever the points were.
//
// THE TRANSLATION IS SMALL ON PURPOSE.  The basis difference is two ULPs of a column
// entry times the extent, i.e. about one ULP of w; it only reaches a floor/ceil where
// lc + w nearly CANCELS, and it only survives the add if lc itself sits on a fine
// lattice.  lc is the sum of two rotated vectors whose magnitudes follow |centre| and
// |p|, so a field placed near the origin is what makes the row-0 and column-0 faces
// (lMax.x = 0, lMax.z = 0) usable as knife edges.  The height axis needs no such help:
// its 1/0.02 scale magnifies everything by 50.
// ---------------------------------------------------------------------------
inline Placement rotatedPlacement()
{
    return Placement{ physx::PxQuat(0x1.fb4302p-4f, 0x1.19080cp-2f, -0x1.513cdcp-3f, 0x1.e0e7f8p-1f),
                      physx::PxVec3(0.23f, -0.11f, 0.17f),
                      physx::PxVec3(1.3f, 0.02f, 0.85f) };
}

static const uint32_t ROT_NB_ROWS = 6u, ROT_NB_COLS = 7u;
static const int16_t ROT_HEIGHT = 2;

// A flat field at ROT_HEIGHT, every slot solid: every change of the rectangle or of the
// y cull is then a change of the pair set, which is the only thing the kernels emit.
inline std::vector<uint32_t> rotatedBlob()
{
    std::vector<uint32_t> blob(2u + ROT_NB_ROWS * ROT_NB_COLS);
    blob[0] = ROT_NB_ROWS;
    blob[1] = ROT_NB_COLS;
    for (uint32_t i = 0; i < ROT_NB_ROWS * ROT_NB_COLS; ++i)
        blob[2u + i] = uint32_t(uint16_t(ROT_HEIGHT)) | (3u << 16) | (3u << 24);
    return blob;
}

static const uint32_t ROT_TWINS[6] = { OLD_ALL, OLD_CENTRE, OLD_BASIS, OLD_COL0, OLD_COL1, OLD_COL2 };
static const char* const ROT_TWIN_NAMES[6] = {
    "the whole pre-P9.89 chain", "rotateInv(c - p) centre", "rotateInv(e_i) basis",
    "rotateInv(e_x) column 0", "rotateInv(e_y) column 1", "rotateInv(e_z) column 2" };

// `nbPoints` world points: knife-edge points found for every twin, spread over the array at
// a fixed stride so they land in many warps and in both `j` iterations, and the rest a
// deterministic LCG scatter over (and around) the field.  `isKnife[i]` marks the former.
inline void rotatedPoints(const Placement& hf, const Field& f, float radius, uint32_t nbPoints,
                          std::vector<physx::PxVec3>& pts, std::vector<uint8_t>& isKnife)
{
    using namespace physx;
    struct Knife { uint32_t axis; bool useMax; float k; float row, col; };
    std::vector<Knife> knives;
    for (int j = 0; j < 5; ++j) knives.push_back({ 0u, true, 0.f, 0.f, 0.5f + float(j) });     // row-0 face
    for (int j = 0; j < 4; ++j) knives.push_back({ 2u, true, 0.f, 0.5f + float(j), 0.f });     // column-0 face
    for (int j = 0; j < 3; ++j)                                                             // con1: above
        knives.push_back({ 1u, false, float(ROT_HEIGHT + 1), 0.5f + float(j), 1.5f + float(j) });
    for (int j = 0; j < 3; ++j)                                                             // con0: below
        knives.push_back({ 1u, true, float(ROT_HEIGHT), 1.5f + float(j), 0.5f + float(j) });

    std::vector<PxVec3> found;
    for (uint32_t t = 0; t < 6u; ++t)
        for (const Knife& k : knives)
        {
            PxVec3 p;
            if (findKnife(hf, f, radius, k.axis, k.useMax, k.k, PxVec3(k.row, float(ROT_HEIGHT), k.col),
                          ROT_TWINS[t], p))
                found.push_back(p);
        }

    pts.assign(nbPoints, PxVec3(0.f));
    isKnife.assign(nbPoints, 0u);
    uint32_t s = 0x2545f491u;
    auto u01 = [&]() { s = s * 1664525u + 1013904223u; return float(s >> 8) * (1.0f / 16777216.0f); };
    for (uint32_t i = 0; i < nbPoints; ++i)
    {
        const float row = -0.4f + u01() * 5.8f;
        const float col = -0.4f + u01() * 6.8f;
        const float h = -10.0f + u01() * 24.0f;
        pts[i] = toWorld(hf, PxVec3(row, h, col));
    }
    const uint32_t stride = found.empty() ? 1u : nbPoints / uint32_t(found.size());
    for (size_t i = 0; i < found.size(); ++i)
    {
        const uint32_t at = uint32_t(i) * stride + (uint32_t(i) * 7u) % stride;
        pts[at] = found[i];
        isKnife[at] = 1u;
    }
}

} // namespace hflb
} // namespace pxgtest

#endif // PXG_HF_LOCAL_BOUND_REF_H
