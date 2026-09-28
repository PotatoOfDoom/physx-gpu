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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS \'\'AS IS\'\' AND ANY
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
//
// ===========================================================================
// pxgSlangTriBox — Gu::intersectTriangleBox_RefImpl<true>, once.
//
// Extracted from npMidphaseCvx at P8.24, when the convex-vs-trimesh midphase
// was about to become a second consumer on the BDA side.  The same move P8.15
// made for the cloth midphase and P8.6 for the GJK/EPA drivers, for the reason
// this port has already paid for once: a mutant that deleted an OUTER gate
// survived because a DUPLICATED inner copy still culled (see
// softBodyClothMidphaseBda\'s header).  One copy, one place to mutate.
//
// Ground truth: geomutils/include/GuIntersectionTriangleBoxRef.h, the
// Akenine-Moller separating-axis test.  Thirteen axes in the CUDA\'s exact
// evaluation ORDER — three early vertex-in-box checks, nine edge-cross axes,
// three AABB axes, then the triangle plane.
//
// TWO DETAILS THAT SURVIVE ONLY IF TRANSCRIBED LITERALLY:
//
//   - planeBoxOverlap is ASYMMETRIC.  The near-corner test rejects on
//     `> 0.0f` and the far-corner test accepts on `>= 0.0f`
//     (GuIntersectionTriangleBoxRef.h:99-104).  A triangle plane exactly
//     touching the far corner counts as an overlap; one exactly touching the
//     near corner does not.  Making the two match either way changes the
//     emitted triangle set on any exactly-tangent plane.
//
//   - the vertex checks are an early-out for the WHOLE conjunction, not one
//     more axis.  They can only fire where the conjunction would also succeed
//     (a vertex inside the box implies overlap), so they change no result — but
//     they DO change which floating-point expressions get evaluated, and under
//     NaN input they are the only path that returns true without touching the
//     axis tests.
//
// CONTRACTION.  Every AXISTEST is literally `p = a*v.y - b*v.z` compared against
// `rad = fa*extents.y + fb*extents.z` — mul feeding add, the exact shape
// contraction moves.  npMidphaseCvx, the Phase-1 harness consumer, is NOT in
// PXG_NOCONTRACT_SHADERS; the BDA consumer that follows WILL be, as every other
// mesh midphase in this port is.  The two therefore compile this same source to
// different arithmetic on near-tangent triangles, and that is deliberate rather
// than an oversight: the stamp is a property of the .spv, not of the module, and
// the engine-path shader is the one whose float behaviour has to match nvcc\'s
// contracted reference.
// ===========================================================================

module pxgSlangTriBox;

// P9.86f: every AXISTEST_* and FINDMINMAX of GuIntersectionTriangleBoxRef.h:58-61/108-156 is
// physx::intrinsics::selectMin/selectMax (PxUnixMathIntrinsics.h:116-125) = a<b?a:b / a>b?a:b, in
// that operand order -- NOT fminf/fmaxf.  GLSL min/max (FMin/FMax, minNum/maxNum on llvmpipe) drop a
// NaN operand, where the select returns the SECOND operand whenever the compare is false: with the
// second operand NaN the CUDA gets minimum = maximum = NaN, both `minimum>rad || maximum<-rad`
// compares are false and the axis does not reject, where FMin/FMax returned the first operand and
// could reject.  Overflowing world coordinates (|coord| ~ 1e19..1e20, a*v.y and b*v.z both +-inf)
// reach it; the +-0 tie is irrelevant here (every consumer is a sign-blind compare).
float tbSelMin(float a, float b) { return (a < b) ? a : b; }   // intrinsics::selectMin
float tbSelMax(float a, float b) { return (a > b) ? a : b; }   // intrinsics::selectMax

public bool tbPlaneBoxOverlap(float3 normal, float d, float3 maxbox)
{
    float3 vmin, vmax;
    if (normal.x > 0.0f) { vmin.x = -maxbox.x; vmax.x = maxbox.x; }
    else { vmin.x = maxbox.x; vmax.x = -maxbox.x; }
    if (normal.y > 0.0f) { vmin.y = -maxbox.y; vmax.y = maxbox.y; }
    else { vmin.y = maxbox.y; vmax.y = -maxbox.y; }
    if (normal.z > 0.0f) { vmin.z = -maxbox.z; vmax.z = maxbox.z; }
    else { vmin.z = maxbox.z; vmax.z = -maxbox.z; }

    if (dot(normal, vmin) + d > 0.0f)
        return false;
    if (dot(normal, vmax) + d >= 0.0f)
        return true;
    return false;
}

// The 9 edge-cross axis tests + 3 AABB axes + plane test, vertex checks
// enabled, in the CUDA's exact evaluation order.
public bool tbIntersectTriangleBox(float3 boxcenter, float3 extents, float3 tp0, float3 tp1, float3 tp2)
{
    const float3 v0 = tp0 - boxcenter;
    const float3 v1 = tp1 - boxcenter;
    const float3 v2 = tp2 - boxcenter;

    // bDoVertexChecks
    if (abs(v0.x) <= extents.x && abs(v0.y) <= extents.y && abs(v0.z) <= extents.z)
        return true;
    if (abs(v1.x) <= extents.x && abs(v1.y) <= extents.y && abs(v1.z) <= extents.z)
        return true;
    if (abs(v2.x) <= extents.x && abs(v2.y) <= extents.y && abs(v2.z) <= extents.z)
        return true;

    const float3 e0 = v1 - v0;
    const float3 e1 = v2 - v1;
    const float3 e2 = v0 - v2;

    float minimum, maximum, rad, p0, p1, p2;

    float fex = abs(e0.x);
    float fey = abs(e0.y);
    float fez = abs(e0.z);
    // AXISTEST_X01(e0.z, e0.y, fez, fey)
    p0 = e0.z * v0.y - e0.y * v0.z;
    p2 = e0.z * v2.y - e0.y * v2.z;
    minimum = tbSelMin(p0, p2); maximum = tbSelMax(p0, p2);
    rad = fez * extents.y + fey * extents.z;
    if (minimum > rad || maximum < -rad) return false;
    // AXISTEST_Y02(e0.z, e0.x, fez, fex)
    p0 = -e0.z * v0.x + e0.x * v0.z;
    p2 = -e0.z * v2.x + e0.x * v2.z;
    minimum = tbSelMin(p0, p2); maximum = tbSelMax(p0, p2);
    rad = fez * extents.x + fex * extents.z;
    if (minimum > rad || maximum < -rad) return false;
    // AXISTEST_Z12(e0.y, e0.x, fey, fex)
    p1 = e0.y * v1.x - e0.x * v1.y;
    p2 = e0.y * v2.x - e0.x * v2.y;
    minimum = tbSelMin(p1, p2); maximum = tbSelMax(p1, p2);
    rad = fey * extents.x + fex * extents.y;
    if (minimum > rad || maximum < -rad) return false;

    fex = abs(e1.x);
    fey = abs(e1.y);
    fez = abs(e1.z);
    // AXISTEST_X01(e1.z, e1.y, fez, fey)
    p0 = e1.z * v0.y - e1.y * v0.z;
    p2 = e1.z * v2.y - e1.y * v2.z;
    minimum = tbSelMin(p0, p2); maximum = tbSelMax(p0, p2);
    rad = fez * extents.y + fey * extents.z;
    if (minimum > rad || maximum < -rad) return false;
    // AXISTEST_Y02(e1.z, e1.x, fez, fex)
    p0 = -e1.z * v0.x + e1.x * v0.z;
    p2 = -e1.z * v2.x + e1.x * v2.z;
    minimum = tbSelMin(p0, p2); maximum = tbSelMax(p0, p2);
    rad = fez * extents.x + fex * extents.z;
    if (minimum > rad || maximum < -rad) return false;
    // AXISTEST_Z0(e1.y, e1.x, fey, fex)
    p0 = e1.y * v0.x - e1.x * v0.y;
    p1 = e1.y * v1.x - e1.x * v1.y;
    minimum = tbSelMin(p0, p1); maximum = tbSelMax(p0, p1);
    rad = fey * extents.x + fex * extents.y;
    if (minimum > rad || maximum < -rad) return false;

    fex = abs(e2.x);
    fey = abs(e2.y);
    fez = abs(e2.z);
    // AXISTEST_X2(e2.z, e2.y, fez, fey)
    p0 = e2.z * v0.y - e2.y * v0.z;
    p1 = e2.z * v1.y - e2.y * v1.z;
    minimum = tbSelMin(p0, p1); maximum = tbSelMax(p0, p1);
    rad = fez * extents.y + fey * extents.z;
    if (minimum > rad || maximum < -rad) return false;
    // AXISTEST_Y1(e2.z, e2.x, fez, fex)
    p0 = -e2.z * v0.x + e2.x * v0.z;
    p1 = -e2.z * v1.x + e2.x * v1.z;
    minimum = tbSelMin(p0, p1); maximum = tbSelMax(p0, p1);
    rad = fez * extents.x + fex * extents.z;
    if (minimum > rad || maximum < -rad) return false;
    // AXISTEST_Z12(e2.y, e2.x, fey, fex)
    p1 = e2.y * v1.x - e2.x * v1.y;
    p2 = e2.y * v2.x - e2.x * v2.y;
    minimum = tbSelMin(p1, p2); maximum = tbSelMax(p1, p2);
    rad = fey * extents.x + fex * extents.y;
    if (minimum > rad || maximum < -rad) return false;

    // AABB axes: FINDMINMAX (GuIntersectionTriangleBoxRef.h:58-61), selects in the CUDA's nesting.
    // (FMin/FMax here are equivalent in the RESULT: they differ only for a NaN v1/v2 lane, whose edges
    // make two normal lanes and d NaN, so the CUDA's plane test below returns false as well.)
    minimum = tbSelMin(tbSelMin(v0.x, v1.x), v2.x); maximum = tbSelMax(tbSelMax(v0.x, v1.x), v2.x);
    if (minimum > extents.x || maximum < -extents.x) return false;
    minimum = tbSelMin(tbSelMin(v0.y, v1.y), v2.y); maximum = tbSelMax(tbSelMax(v0.y, v1.y), v2.y);
    if (minimum > extents.y || maximum < -extents.y) return false;
    minimum = tbSelMin(tbSelMin(v0.z, v1.z), v2.z); maximum = tbSelMax(tbSelMax(v0.z, v1.z), v2.z);
    if (minimum > extents.z || maximum < -extents.z) return false;

    // plane of the triangle
    const float3 normal = cross(e0, e1);
    const float d = -dot(normal, v0);
    if (!tbPlaneBoxOverlap(normal, d, extents))
        return false;

    return true;
}

