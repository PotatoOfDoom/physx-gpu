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

// femClothClothContactBda — P7.2 tranche 2: the cloth-vs-CLOTH self-contact SOLVE
// kernels on the Vulkan engine ABI (BDA).  These are what turn the VT/EE self-contacts
// the tranche-1 midphase (femClothSelfCollisionBda) emitted into an actual cloth-vertex
// push so the cloth FOLDS on itself instead of self-intersecting.  Ported line-for-line
// from
//   gpusimulationcontroller/src/CUDA/FEMClothConstraintPrep.cu (the prepare, :155)
//   gpusimulationcontroller/src/CUDA/FEMClothExternalSolve.cu   (the query + solve, :902
//     / :964 / :651 / :760 + the updateContactValidity book-keeping :1114)
// with the closest-point + constraint leaves from gpunarrowphase deformableCollision.cuh
// and geomutils GuDistancePointTriangle.h:
//   - cloth_clothContactPrepareLaunch (FEMClothConstraintPrep.cu:155, launch
//     PxgFEMClothCore.cpp:499, CLOTH_CLOTH_CONTACTPREPARE): clamps the cloth-cloth contact
//     count, then for every not-yet-valid contact recomputes the EXACT rest-pose distance
//     (closestPtEdgeEdge for EE, Gu::closestPtPointTriangle2 for VT) and marks the pair
//     valid when the rest distance clears the self-collision filter — the deferred filter
//     the midphase left "No filtering applied yet".
//   - cloth_queryClothClothContact{VT,EE}CountLaunch (FEMClothExternalSolve.cu:902 / :964,
//     launch PxgFEMClothCore.cpp:1981/:1982, CLOTH_QUERY_CLOTH_CONTACT_{VT,EE}_COUNT): a
//     count pass (countContact) that, for each valid + still-penetrating (distSq <=
//     restDist^2) self-contact, marks it in-collision and accumulates each referenced
//     cloth vertex's reference count into mDeltaPos.w (so applyExternalDeltas can
//     mass-average).  Contacts that separated are marked NOT-in-collision so the solve
//     skips them.
//   - cloth_solveClothClothDelta{VT,EE}Launch (FEMClothExternalSolve.cu:651 / :760, launch
//     PxgFEMClothCore.cpp:2003/:2004, CLOTH_SOLVE_CLOTH_{VT,EE}_COLLISION): solves each
//     in-collision self-contact (triangleCollisionConstraint) and scatters BOTH endpoints'
//     position corrections into mDeltaPos — the vertex + the triangle's three vertices
//     (VT), or edge0's two + edge1's two vertices (EE) — folded next by the already-mapped
//     cloth_applyExternalDeltasLaunch.
//   - cloth_updateClothContactValidityLaunch (FEMClothExternalSolve.cu:1114, launch
//     PxgFEMClothCore.cpp:647, CLOTH_UPDATE_CLOTH_CONTACT_VALIDITY): per-vertex book-keeping
//     that (on an adaptive update) resets the VT/EE contact counts and snapshots each
//     vertex's current position into mPrevPositionInContactOffset (with w = 1 marking it
//     set) so the next contact refresh can measure how far vertices drifted.
//
// SELF SCOPE.  All four launches are the generic cloth-vs-cloth path; on the self-
// collision launch clothId0 == clothId1 (both index the same active cloth), so cloth0 and
// cloth1 alias the same descriptor.  The kernels are transcribed GENERICALLY (they read
// cloth0 / cloth1 by their compressed ids exactly as the CUDA does) — no self-only fold is
// needed here because the generic read already resolves to the same cloth for a self pair,
// and the different-cloth pairs flow through the identical arithmetic.
//
// DOCTRINE (identical to the prior cloth BDA tranches femClothRigidContactBda /
// femClothSelfCollisionBda):
//   - the PxgFEMCloth descriptor arrives as a device-pointer ARRAY read through Ptr<> (the
//     byte-exact pxgSlangFEMClothDesc mirror): mPosition_InvMass @16, mRestPosition @24,
//     mPrevPositionInContactOffset @32, mDynamicFrictions @48, mTriangleVertexIndices @72,
//     mDeltaPos @248, mSelfCollisionFilterDistance @312, mRestDistance @384, mNbVerts @316,
//     mElementIndex-adjacent scalars are the fields these kernels touch;
//   - the 16-byte PxgFemFemContactInfo (PxgFEMCore.h:51: pairInd0 u64 @0, pairInd1 u32 @8,
//     auxInd u32 @12) is addressed by explicit byte offset through Ptr<> (no struct block
//     copy), exactly as femClothRigidContactBda read PxgFemOtherContactInfo.  The auxInd
//     bit packing (bit31 Valid, bit30 PairType EE, bit29 InCollision, bits14-28 AuxInd1,
//     bits0-13 AuxInd0) is reproduced by the same mask/shift arithmetic the header inlines.
//
// DEVIATIONS from the CUDA original (recipe policy; each preserves the constraint
// semantics for a cloth self-contact and is exercised bit-exact by testFemClothSim.cpp):
//   - cloth.mDeltaPos accumulation uses REAL atomics (pxgSlangAtomicFloat), exactly as
//     the CUDA atomicAdd(&mDeltaPos[v].w, coeff) / atomicAdd(&mDeltaPos[v].x/.y/.z, dx).
//     This was previously a plain load-add-store, justified by "a single self-contact
//     touches four DISTINCT vertices".  That is true PER CONTACT and says nothing about
//     ACROSS contacts, which is where the contention is: in a folded cloth a vertex takes
//     part in many self-contacts at once.  Measured consequence, see pxgSlangAtomicFloat;
//   - the PxgFemFemContactInfo reference writes (markValid in the prepare, markInCollision
//     in the count) touch ONLY the 4-byte auxInd word (@12).  In CUDA the prepare copies
//     the struct, flips a bit, and writes the whole 16 bytes back (pairInd0/pairInd1
//     unchanged), and the count kernel mutates the auxInd of a reference in place — both
//     are byte-identical to writing just the auxInd word;
//   - FRICTION is the null-material path.  applyContactDelta reads the per-vertex
//     mDynamicFrictions VERBATIM and forms the same 0.25*(f0+f1+f2+f3) combined
//     coefficient, and triangleCollisionConstraint builds the full tangent constraint —
//     but its friction branch is gated `frictionCoefficient != 0`.  With zero cloth
//     friction (the self-fold / drape target and the unit test) the branch is DEAD, so no
//     value flows through PxRecipSqrt and the solve is FULLY bit-exact.  When friction IS
//     non-zero that branch's PxRecipSqrt (a hardware reciprocal-sqrt, NOT bitwise
//     1/sqrt) is a DOCUMENTED TOLERANCE seam — a follow-on material tranche;
//   - the clothMaterials arg (prepare) and the dt arg (both solves) and the
//     forceUpdateClothContactPairs + dt args (updateContactValidity) are carried in the
//     push blob for ABI parity but NOT read: the CUDA kernel bodies never reference them.
//
// FLOAT LOWERING.  P9.36 CORRECTED THIS, AND THE OLD RULE IS WORTH RECORDING BECAUSE IT
// IS WHY THE DEFECT WAS EVERYWHERE.  This header used to say:
//
//     "a PxVec3/float scalar divide is transcribed as Slang `float3 / float`
//      (per-component OpFDiv), and an explicit PxVec3 * (1/x) as reciprocal-multiply —
//      each is mirrored bit-for-bit by the CPU reference."
//
// Both halves lower to the SAME thing in the engine.  PxVec3::operator/ (PxVec3.h:219) is
// `f = Type(1.0)/f; return PxVec3T(x*f, y*f, z*f);` -- a reciprocal and three multiplies --
// and so are operator/= (:260), normalize() (:295) and the CUDA float3/float4 operators the
// GPU kernels use (cutil_math.h:342/352/485/495).  There is no per-component form to
// transcribe.  This very kernel's own source shows it: FEMClothExternalSolve.cu:748 writes
// `normal = disp / dist;` and :846 writes `normal = disp * (1.0f / dist);`, on the same
// type for the same purpose, because they ARE the same operation.  The port read them as
// two lowerings and made the difference a rule.
//
// "each is mirrored bit-for-bit by the CPU reference" was the justification, and it is the
// failure mode rather than the proof: the mirror was written from this convention, so
// shader and mirror agreed with each other while both diverged from PhysX by >=1 component
// on 53.4595% of vectors (measured, 3e6 samples).  gpuslang.vectorDivide now checks the
// SPIR-V directly, which is the only thing here that can.
//
// The rule is: a vector divided by a SCALAR is one reciprocal and three multiplies.  A
// vector divided by a VECTOR is genuinely componentwise (cutil_math.h:337) and stays a
// divide.  All float arithmetic is
// NoContraction-stamped (PXG_NOCONTRACT_SHADERS): the dot/cross/normalize-heavy closest-
// point + constraint math (disp.dot(normal), the response denominator, the bias CN, the
// deltaLambda scatter) feeds the compared mDeltaPos and would fuse to FMAs otherwise,
// breaking bitwise parity with the -ffp-contract=off CPU reference (same reasoning as
// femClothRigidContactBda).  There is NO transcendental on the live (frictionless) path
// (pure mul/add/sub/div/sqrt/min/max/abs) — this tranche is FULLY bit-exact, no tolerance
// seam in scope.

import pxgSlangAtomicFloat;   // atomicAdd(float*) parity — see that module
import pxgSlangFEMClothDesc;   // PxgFEMClothGpu (mPosition_InvMass@16, mDeltaPos@248, ...)

// PxgSoftBodyKernelBlockDim/GridDim::SB_UPDATEROTATION (PxgSoftBodyCoreKernelIndices.h:42/59)
// — the prepare / query / solve grid+block; PxgFEMClothKernelBlockDim::CLOTH_STEP
// (PxgFEMClothCoreKernelIndices.h:40) — the updateContactValidity block.
static const uint RC_BLOCK = 256u;
static const uint RC_GRID = 1024u;
static const uint CLOTH_STEP = 1024u;

// PX_MAX_NB_DEFORMABLE_SURFACE_TRI (PxDeformableSurface.h:56) == the element mask;
// PXG_BITSHIFT_ELEMENT_ID (PxgSimulationCoreDesc.h:41) — cloth id / elem index packing
// (PxGetClothId / PxGetClothElementIndex, PxgSimulationCoreDesc.h:135/140).
static const uint PX_MAX_NB_DEFORMABLE_SURFACE_TRI = 0x000fffffu;
static const uint PXG_CLOTH_ID_SHIFT = 20u;

// FEMCLOTH_THRESHOLD (FEMClothUtil.cuh:65).
static const float FEMCLOTH_THRESHOLD = 1.0e-14f;

// deformableCollision.cuh:37/38.
static const float DEFORMABLE_BARYCENTRIC_THRESHOLD = 1.0e-6f;
static const float DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD = 1.0f - 1.0e-6f;

// PxgFemFemContactInfo (PxgFEMCore.h:51, PX_ALIGN(16), 16 bytes): pairInd0 u64 @0,
// pairInd1 u32 @8, auxInd u32 @12.  auxInd bit packing (PxgFEMCore.h:69-82).
static const uint CI_pairInd0 = 0u;
static const uint CI_pairInd1 = 8u;
static const uint CI_auxInd = 12u;
static const uint CI_STRIDE = 16u;

static const uint AUX_IND0_BITS = 14u;
static const uint AUX_IND1_BITS = 15u;
static const uint AUX_IND0_MASK = (1u << AUX_IND0_BITS) - 1u;   // 0x00003FFF
static const uint AUX_IND1_SHIFT = AUX_IND0_BITS;
static const uint CONTACT_FLAG_MASK   = 1u << 29;              // isInCollision
static const uint PAIR_TYPE_FLAG_MASK  = 1u << 30;             // 0 = VT, 1 = EE
static const uint VALIDITY_FLAG_MASK   = 1u << 31;

// ---------------------------------------------------------------------------
// PxgFemFemContactInfo byte-offset readers / writers + auxInd bit helpers.
// ---------------------------------------------------------------------------
uint ciPairInd0Lo(uint64_t ciBase) { return Ptr<uint>(ciBase + CI_pairInd0)[0]; }   // PxU32(pairInd0)
uint ciPairInd1(uint64_t ciBase)   { return Ptr<uint>(ciBase + CI_pairInd1)[0]; }
uint ciAuxInd(uint64_t ciBase)     { return Ptr<uint>(ciBase + CI_auxInd)[0]; }
void ciWriteAuxInd(uint64_t ciBase, uint aux) { Ptr<uint>(ciBase + CI_auxInd)[0] = aux; }

bool auxIsValidPair(uint aux)     { return (aux & VALIDITY_FLAG_MASK) != 0u; }
bool auxIsEdgeEdgePair(uint aux)  { return (aux & PAIR_TYPE_FLAG_MASK) != 0u; }
bool auxIsInCollision(uint aux)   { return (aux & CONTACT_FLAG_MASK) != 0u; }
uint auxGetInd0(uint aux)         { return aux & AUX_IND0_MASK; }
uint auxGetInd1(uint aux)         { return (aux >> AUX_IND1_SHIFT) & ((1u << AUX_IND1_BITS) - 1u); }
uint auxMarkValid(uint aux)       { return aux | VALIDITY_FLAG_MASK; }
uint auxMarkInCollision(uint aux, bool inContact)
{
    return (aux & ~CONTACT_FLAG_MASK) | (inContact ? CONTACT_FLAG_MASK : 0u);
}

uint pxGetClothId(uint compressedId)      { return compressedId >> PXG_CLOTH_ID_SHIFT; }
uint pxGetClothElementIndex(uint compressedId) { return compressedId & PX_MAX_NB_DEFORMABLE_SURFACE_TRI; }

// Index x/y/z of a uint4 by a runtime edge index in {0,1,2}
// (CUDA reinterpret_cast<const PxU32*>(&triVertInd)[localIndex]).
uint pick3(uint4 v, uint i) { return (i == 0u) ? v.x : ((i == 1u) ? v.y : v.z); }

// ---------------------------------------------------------------------------
// Geometry leaves (byte-exact transcriptions).
// ---------------------------------------------------------------------------

// closestPtPointTriangle (deformableCollision.cuh:146) — returns closest point, s/t out.
float3 closestPtPointTriangle(float3 p, float3 a, float3 b, float3 c, out float s, out float t)
{
    const float3 ab = b - a;
    const float3 ac = c - a;
    const float3 ap = p - a;

    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) { s = 0.0f; t = 0.0f; return a; }

    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) { s = 1.0f; t = 0.0f; return b; }

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        s = v; t = 0.0f;
        return a + v * ab;
    }

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) { s = 0.0f; t = 1.0f; return c; }

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        s = 0.0f; t = w;
        return a + w * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        s = 1.0f - w; t = w;
        return b + w * (c - b);
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    s = v; t = w;
    return a + ab * v + ac * w;
}

// Gu::closestPtPointTriangle2 (GuDistancePointTriangle.h:44) — ab / ac precomputed.  The
// EXACT rest-distance filter the prepare uses for VT self-contacts.
float3 closestPtPointTriangle2(float3 p, float3 a, float3 b, float3 c, float3 ab, float3 ac)
{
    const float3 ap = p - a;
    const float d1 = dot(ab, ap);
    const float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    const float3 bp = p - b;
    const float d3 = dot(ab, bp);
    const float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        const float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    const float3 cp = p - c;
    const float d5 = dot(ab, cp);
    const float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        const float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return a + ab * v + ac * w;
}

// closestPtLineLine (deformableCollision.cuh:244) — infinite-line closest points; returns
// false for a parallel / degenerate pair.  P9.36: `const PxVec3 u0 = r0 / r0Len;` (:265) is
// PxVec3::operator/, i.e. a reciprocal and three multiplies; this comment used to claim it
// was per-component division "matching the p72ref transcription", which is the convention
// corrected in the FLOAT LOWERING note above.
bool closestPtLineLine(float3 e0v0, float3 e0v1, float3 e1v0, float3 e1v1, out float s, out float t, out float distSq)
{
    const float3 r0 = e0v1 - e0v0;
    const float3 r1 = e1v1 - e1v0;
    const float3 d = e1v0 - e0v0;

    const float r0Len2 = dot(r0, r0);
    const float r1Len2 = dot(r1, r1);

    if (r0Len2 < 1.0e-18f || r1Len2 < 1.0e-18f)
    {
        s = 0.0f; t = 0.0f;
        distSq = dot(d, d);
        return false;
    }

    const float r0Len = sqrt(r0Len2);
    const float r1Len = sqrt(r1Len2);

    // P9.36: deformableCollision.cuh:265-266 are `const PxVec3 u0 = r0 / r0Len;` --
    // PxVec3::operator/ (PxVec3.h:219) is ONE reciprocal and three multiplies.  The two
    // scalar divides further down (b0, b1) are scalar/scalar and stay divides.
    const float3 u0 = r0 * (1.0f / r0Len);
    const float3 u1 = r1 * (1.0f / r1Len);

    const float cosTheta = abs(dot(u0, u1));

    if (cosTheta > 0.9999f)
    {
        const float tProj = dot(e0v0 - e1v0, r1) / r1Len2;
        t = tProj;
        s = 0.0f;

        const float3 pt0 = e0v0;
        const float3 pt1 = e1v0 + r1 * tProj;
        distSq = dot(pt1 - pt0, pt1 - pt0);
        return false;
    }

    const float a01 = -dot(u0, u1);
    const float b0 = dot(u0, d) / r0Len;
    const float b1 = -dot(u1, d) / r1Len;

    const float det = 1.0f - a01 * a01;

    if (det < 1.0e-9f)
    {
        s = 0.0f; t = 0.0f;
        distSq = dot(d, d);
        return false;
    }

    const float invDet = 1.0f / det;
    s = (b0 - b1 * a01) * invDet;
    t = (b1 - a01 * b0) * invDet;

    const float3 pt0 = e0v0 + s * r0;
    const float3 pt1 = e1v0 + t * r1;
    distSq = dot(pt1 - pt0, pt1 - pt0);

    return true;
}

// closestPtPointEdge (deformableCollision.cuh:231) — the closestPtEdgeEdge segment
// fallback leaf.  PxClamp(x,lo,hi) == min(hi, max(lo, x)).
float closestPtPointEdge(float3 v0, float3 ev0, float3 ev1)
{
    const float3 r = ev1 - ev0;
    const float d2 = dot(r, r);
    return (d2 > 0.0f) ? min(1.0f, max(0.0f, dot(r, v0 - ev0) / d2)) : 0.5f;
}

// closestPtEdgeEdge (deformableCollision.cuh:313) — finite-edge closest points; the EXACT
// rest-distance filter the prepare uses for EE self-contacts.  Falls back to the point-edge
// candidates when the infinite-line solution leaves the [0,1] x [0,1] box.
bool closestPtEdgeEdge(float3 e0v0, float3 e0v1, float3 e1v0, float3 e1v1, out float s, out float t, out float distSq)
{
    if (!closestPtLineLine(e0v0, e0v1, e1v0, e1v1, s, t, distSq))
        return false;

    if (s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f)
    {
        float w[4];
        w[0] = closestPtPointEdge(e0v0, e1v0, e1v1);
        w[1] = closestPtPointEdge(e0v1, e1v0, e1v1);
        w[2] = closestPtPointEdge(e1v0, e0v0, e0v1);
        w[3] = closestPtPointEdge(e1v1, e0v0, e0v1);

        float2 candidates[4];
        candidates[0] = float2(0.0f, w[0]);
        candidates[1] = float2(1.0f, w[1]);
        candidates[2] = float2(w[2], 0.0f);
        candidates[3] = float2(w[3], 1.0f);

        int minIndex = 0;
        distSq = 3.4028234663852886e+38f;  // PX_MAX_REAL
        for (int i = 0; i < 4; ++i)
        {
            const float sC = candidates[i].x;
            const float tC = candidates[i].y;
            const float3 p0 = e0v0 * (1.0f - sC) + e0v1 * sC;
            const float3 p1 = e1v0 * (1.0f - tC) + e1v1 * tC;
            const float d2 = dot(p0 - p1, p0 - p1);
            minIndex = (d2 < distSq) ? i : minIndex;
            distSq = min(d2, distSq);
        }

        s = candidates[minIndex].x;
        t = candidates[minIndex].y;
    }

    return true;
}

// triangleCollisionConstraint (FEMClothExternalSolve.cu:243) — VT + EE shared constraint.
// invMass here is the count-split inverse mass.  Friction branch DEAD for zero
// frictionCoefficient (the frictionless self-fold / unit-test path); see header.  Returns
// the four per-endpoint deltas (0 when the response denominator is degenerate).
bool triangleCollisionConstraint(out float3 dx0, out float3 dx1, out float3 dx2, out float3 dx3,
                                 float3 disp, float3 normal, float3 relDelta, float4 invMass,
                                 float restDistance, float frictionCoefficient, float4 weights)
{
    dx0 = float3(0.0f, 0.0f, 0.0f);
    dx1 = float3(0.0f, 0.0f, 0.0f);
    dx2 = float3(0.0f, 0.0f, 0.0f);
    dx3 = float3(0.0f, 0.0f, 0.0f);

    const float CN = dot(disp, normal) - restDistance;

    const float denom = invMass.x * weights.x * weights.x + invMass.y * weights.y * weights.y +
                        invMass.z * weights.z * weights.z + invMass.w * weights.w * weights.w;
    if (denom < FEMCLOTH_THRESHOLD)
        return false;

    const float invDenom = 1.0f / denom;
    const float deltaLambdaN = -CN * invDenom;

    float3 commonVec = deltaLambdaN * normal;

    // Friction constraint in the tangent direction (DEAD for frictionCoefficient == 0).
    float3 tanDir = relDelta - dot(relDelta, normal) * normal;
    const float tanMagSq = dot(tanDir, tanDir);

    if (frictionCoefficient != 0.0f && tanMagSq > FEMCLOTH_THRESHOLD)
    {
        const float invTanMag = 1.0f / sqrt(tanMagSq);   // PxRecipSqrt (documented seam; DEAD here)
        tanDir = tanDir * invTanMag;
        const float CT = dot(relDelta, tanDir);

        float deltaLambdaT = max(0.0f, CT * invDenom);
        deltaLambdaT = -min(deltaLambdaT, frictionCoefficient * abs(deltaLambdaN));

        commonVec = commonVec + deltaLambdaT * tanDir;
    }

    dx0 = invMass.x * weights.x * commonVec;
    dx1 = invMass.y * weights.y * commonVec;
    dx2 = invMass.z * weights.z * commonVec;
    dx3 = invMass.w * weights.w * commonVec;

    return true;
}

// mDeltaPos accumulators — REAL atomics (pxgSlangAtomicFloat), matching the CUDA
// atomicAdd.  These used to be a plain load-add-store on the argument that a single
// contact touches four DISTINCT vertices; that is true per contact and irrelevant
// across contacts, which is where the contention actually is.  See the module header.
void deltaBumpW(uint64_t deltaBase, uint vertIdx, float coeff)
{
    pxgAtomicAddF4W(deltaBase, vertIdx, coeff);
}
void deltaAddXYZ(uint64_t deltaBase, uint vertIdx, float3 dx)
{
    pxgAtomicAddF4XYZ(deltaBase, vertIdx, dx);
}

// applyContactDelta (FEMClothExternalSolve.cu:605) — the shared VT/EE scatter.  Per
// endpoint we pass its cloth's mDeltaPos base (dp*) and mDynamicFrictions base (df*): for
// VT ep0 -> cloth0, ep1/2/3 -> cloth1; for EE ep0/1 -> cloth0, ep2/3 -> cloth1.
void applyContactDelta(uint64_t dp0, uint64_t dp1, uint64_t dp2, uint64_t dp3,
                       uint64_t df0, uint64_t df1, uint64_t df2, uint64_t df3,
                       uint4 vi, float3 disp, float3 normal, float4 invMasses, float4 weights,
                       float3 relDelta, float restDist)
{
    const float count0 = Ptr<float4>(dp0)[vi.x].w;
    const float count1 = Ptr<float4>(dp1)[vi.y].w;
    const float count2 = Ptr<float4>(dp2)[vi.z].w;
    const float count3 = Ptr<float4>(dp3)[vi.w].w;

    const float4 splittedInvMasses = float4(count0 * invMasses.x, count1 * invMasses.y,
                                            count2 * invMasses.z, count3 * invMasses.w);

    const float frictionCoefficient = 0.25f * (Ptr<float>(df0)[vi.x] + Ptr<float>(df1)[vi.y] +
                                               Ptr<float>(df2)[vi.z] + Ptr<float>(df3)[vi.w]);

    float3 dx0, dx1, dx2, dx3;
    triangleCollisionConstraint(dx0, dx1, dx2, dx3, disp, normal, relDelta, splittedInvMasses,
                                restDist, frictionCoefficient, weights);

    if (count0 > 0.0f) deltaAddXYZ(dp0, vi.x, dx0);
    if (count1 > 0.0f) deltaAddXYZ(dp1, vi.y, dx1);
    if (count2 > 0.0f) deltaAddXYZ(dp2, vi.z, dx2);
    if (count3 > 0.0f) deltaAddXYZ(dp3, vi.w, dx3);
}

// countContact (FEMClothExternalSolve.cu:861) — the shared VT/EE count.  cloth0or1 == the
// endpoint-1 cloth (cloth1 for VT, cloth0 for EE).
void countContact(uint64_t ciBase, uint aux, float4 coeffs, float3 xx0, float3 xx1, float3 xx2, float3 xx3,
                  uint64_t dp0, uint64_t dp1, uint64_t dp2, uint64_t dp3, uint4 vi,
                  float restDist)
{
    const float3 disp = coeffs.x * xx0 + coeffs.y * xx1 + coeffs.z * xx2 + coeffs.w * xx3;
    const float distSq = dot(disp, disp);

    if (distSq > restDist * restDist)
    {
        ciWriteAuxInd(ciBase, auxMarkInCollision(aux, false));
        return;
    }

    const float coeff0 = abs(coeffs.x);
    const float coeff1 = abs(coeffs.y);
    const float coeff2 = abs(coeffs.z);
    const float coeff3 = abs(coeffs.w);

    if (coeff0 > 1e-3f) deltaBumpW(dp0, vi.x, coeff0);
    if (coeff1 > 1e-3f) deltaBumpW(dp1, vi.y, coeff1);
    if (coeff2 > 1e-3f) deltaBumpW(dp2, vi.z, coeff2);
    if (coeff3 > 1e-3f) deltaBumpW(dp3, vi.w, coeff3);

    ciWriteAuxInd(ciBase, auxMarkInCollision(aux, true));
}

// ===========================================================================
// Push-constant blobs (byte-identical to the CUDA arg packs / ABI rows).
// ===========================================================================

// cloth_clothContactPrepareLaunch (FEMClothConstraintPrep.cu:155).  CUDA arg pack
// (PxgFEMClothCore.cpp:505): { PxgFEMCloth* clothes, PxgFemFemContactInfo* contactInfos,
// PxU32* numContacts, PxU32 maxContacts, PxsDeformableSurfaceMaterialData* clothMaterials,
// const PxU8* updateContactPairs } -> natural-alignment blob = 48 bytes.
// (ABI row {8,8,8,4,8,8}.)
struct PrepClothClothParams
{
    uint64_t clothes;             // @0
    uint64_t contactInfos;        // @8
    uint64_t numContacts;         // @16
    uint maxContacts;             // @24
    uint _pad28;                  // @28
    uint64_t clothMaterials;      // @32  (unused; ABI parity)
    uint64_t updateContactPairs;  // @40  -> 48
}

// cloth_queryClothClothContact{VT,EE}CountLaunch (FEMClothExternalSolve.cu:902 / :964).
// CUDA arg pack (PxgFEMClothCore.cpp:1983): { PxgFEMCloth* cloths,
// PxgFemFemContactInfo* contactInfos, PxU32* numContacts } -> 24-byte blob.
// (ABI row {8,8,8}.)
struct QueryClothClothParams
{
    uint64_t cloths;         // @0
    uint64_t contactInfos;   // @8
    uint64_t numContacts;    // @16 -> 24
}

// cloth_solveClothClothDelta{VT,EE}Launch (FEMClothExternalSolve.cu:651 / :760).  CUDA arg
// pack (PxgFEMClothCore.cpp:2005): { PxgFEMCloth* cloths, PxgFemFemContactInfo* contactInfos,
// PxU32* numContacts, PxReal dt } -> 32-byte blob.  (ABI row {8,8,8,4}.)
struct SolveClothClothParams
{
    uint64_t cloths;         // @0
    uint64_t contactInfos;   // @8
    uint64_t numContacts;    // @16
    float dt;                // @24  (unused; ABI parity)
    uint _pad28;             // @28 -> 32
}

// cloth_updateClothContactValidityLaunch (FEMClothExternalSolve.cu:1114).  CUDA arg pack
// (PxgFEMClothCore.cpp:652): { PxgFEMCloth* femCloths, const PxU32* activeIds,
// bool adaptiveCollisionPairUpdate, bool forceUpdateClothContactPairs,
// const PxU8* updateContactPairs, PxU32* VTContactCount, PxU32* EEContactCount,
// PxReal dt } -> natural-alignment blob = 56 bytes.  (ABI row {8,8,1,1,8,8,8,4}.)
struct UpdateValidityParams
{
    uint64_t femCloths;                 // @0
    uint64_t activeIds;                 // @8
    uint adaptiveFlagWord;              // @16  bool adaptiveCollisionPairUpdate (byte0)
    uint forceUpdateFlagWord;          // @20  bool forceUpdateClothContactPairs (byte0) (unused)
    uint64_t updateContactPairs;        // @24
    uint64_t VTContactCount;            // @32
    uint64_t EEContactCount;            // @40
    float dt;                           // @48  (unused; ABI parity)
    uint _pad52;                        // @52 -> 56
}

[[vk::push_constant]] ConstantBuffer<PrepClothClothParams> gPrep;
[[vk::push_constant]] ConstantBuffer<QueryClothClothParams> gQuery;
[[vk::push_constant]] ConstantBuffer<SolveClothClothParams> gSolve;
[[vk::push_constant]] ConstantBuffer<UpdateValidityParams> gUpdate;

// ===========================================================================
// cloth_clothContactPrepareLaunch (FEMClothConstraintPrep.cu:155)
// ===========================================================================
[shader("compute")]
[numthreads(RC_BLOCK, 1, 1)] // SB_UPDATEROTATION
void cloth_clothContactPrepareLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    // Early exit if contact pairs are not updated (*updateContactPairs == 0; PxU8).
    if ((Ptr<uint>(gPrep.updateContactPairs)[0] & 0xFFu) == 0u)
        return;

    const uint numContactsRaw = Ptr<uint>(gPrep.numContacts)[0];
    const uint tNumContacts = min(numContactsRaw, gPrep.maxContacts);
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;
    const uint idx = groupThreadId.x;

    Ptr<PxgFEMClothGpu> clothes = Ptr<PxgFEMClothGpu>(gPrep.clothes);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + idx + nbIterationsPerBlock * groupId.x * RC_BLOCK;

        if (workIndex == 0u)   // Clamp the cloth contact count.
            Ptr<uint>(gPrep.numContacts)[0] = tNumContacts;

        if (workIndex >= tNumContacts)
            return;

        const uint64_t ciBase = gPrep.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint aux = ciAuxInd(ciBase);
        if (auxIsValidPair(aux))   // For different cloths, already set valid.
            continue;

        const uint pairInd0 = ciPairInd0Lo(ciBase);
        const uint clothId0 = pxGetClothId(pairInd0);
        const uint elementId0 = pxGetClothElementIndex(pairInd0);

        const uint pairInd1 = ciPairInd1(ciBase);
        const uint clothId1 = pxGetClothId(pairInd1);
        const uint elementId1 = pxGetClothElementIndex(pairInd1);

        const float filterDistance = clothes[clothId0].mSelfCollisionFilterDistance;
        const float filterDistSq = filterDistance * filterDistance;

        if (auxIsEdgeEdgePair(aux))   // Edge-edge collision
        {
            const uint e0_localIndex0 = auxGetInd0(aux);
            const uint e0_localIndex1 = (e0_localIndex0 + 1u) % 3u;
            const uint4 triVertInd0 = Ptr<uint4>(clothes[clothId0].mTriangleVertexIndices)[elementId0];
            const uint e0_v0 = pick3(triVertInd0, e0_localIndex0);
            const uint e0_v1 = pick3(triVertInd0, e0_localIndex1);

            const uint e1_localIndex0 = auxGetInd1(aux);
            const uint e1_localIndex1 = (e1_localIndex0 + 1u) % 3u;
            const uint4 triVertInd1 = Ptr<uint4>(clothes[clothId1].mTriangleVertexIndices)[elementId1];
            const uint e1_v0 = pick3(triVertInd1, e1_localIndex0);
            const uint e1_v1 = pick3(triVertInd1, e1_localIndex1);

            const float3 r0 = Ptr<float4>(clothes[clothId0].mRestPosition)[e0_v0].xyz;
            const float3 r1 = Ptr<float4>(clothes[clothId0].mRestPosition)[e0_v1].xyz;
            const float3 r2 = Ptr<float4>(clothes[clothId1].mRestPosition)[e1_v0].xyz;
            const float3 r3 = Ptr<float4>(clothes[clothId1].mRestPosition)[e1_v1].xyz;

            float s, t, restDistSq;
            closestPtEdgeEdge(r0, r1, r2, r3, s, t, restDistSq);

            if (restDistSq > filterDistSq)
                ciWriteAuxInd(ciBase, auxMarkValid(aux));
        }
        else   // Vertex-triangle collision
        {
            const uint4 triVertId1 = Ptr<uint4>(clothes[clothId1].mTriangleVertexIndices)[elementId1];

            const float3 r0 = Ptr<float4>(clothes[clothId0].mRestPosition)[elementId0].xyz;
            const float3 r1 = Ptr<float4>(clothes[clothId1].mRestPosition)[triVertId1.x].xyz;
            const float3 r2 = Ptr<float4>(clothes[clothId1].mRestPosition)[triVertId1.y].xyz;
            const float3 r3 = Ptr<float4>(clothes[clothId1].mRestPosition)[triVertId1.z].xyz;

            const float3 r12 = r2 - r1;
            const float3 r13 = r3 - r1;

            const float3 closest = closestPtPointTriangle2(r0, r1, r2, r3, r12, r13);
            const float3 diff = r0 - closest;
            const float restDistSq = dot(diff, diff);

            if (restDistSq > filterDistSq)
                ciWriteAuxInd(ciBase, auxMarkValid(aux));
        }
    }
}

// ===========================================================================
// cloth_queryClothClothContactVTCountLaunch (FEMClothExternalSolve.cu:902)
// ===========================================================================
[shader("compute")]
[numthreads(RC_BLOCK, 1, 1)] // SB_UPDATEROTATION
void cloth_queryClothClothContactVTCountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gQuery.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    Ptr<PxgFEMClothGpu> cloths = Ptr<PxgFEMClothGpu>(gQuery.cloths);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + groupThreadId.x + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint64_t ciBase = gQuery.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint aux = ciAuxInd(ciBase);
        if (!auxIsValidPair(aux))
        {
            ciWriteAuxInd(ciBase, auxMarkInCollision(aux, false));
            continue;
        }

        const uint pairInd0 = ciPairInd0Lo(ciBase);
        const uint pairInd1 = ciPairInd1(ciBase);
        const uint clothId0 = pxGetClothId(pairInd0);
        const uint clothId1 = pxGetClothId(pairInd1);
        const uint elementId0 = pxGetClothElementIndex(pairInd0);
        const uint elementId1 = pxGetClothElementIndex(pairInd1);

        const uint4 triVertId1 = Ptr<uint4>(cloths[clothId1].mTriangleVertexIndices)[elementId1];
        const uint4 vertIndices = uint4(elementId0, triVertId1.x, triVertId1.y, triVertId1.z);

        const float3 xx0 = Ptr<float4>(cloths[clothId0].mPosition_InvMass)[vertIndices.x].xyz;
        const float3 xx1 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vertIndices.y].xyz;
        const float3 xx2 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vertIndices.z].xyz;
        const float3 xx3 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vertIndices.w].xyz;

        float s, t;
        const float3 closestPt = closestPtPointTriangle(xx0, xx1, xx2, xx3, s, t);
        const float r = 1.0f - s - t;
        const float4 weights = float4(1.0f, -r, -s, -t);   // cloth0 - cloth1

        const float restDist = cloths[clothId0].mRestDistance + cloths[clothId1].mRestDistance;

        // countContact: cloth0or1 == cloth1 (ep1 belongs to cloth1).
        const uint64_t dp0 = cloths[clothId0].mDeltaPos;
        const uint64_t dp1 = cloths[clothId1].mDeltaPos;
        countContact(ciBase, aux, weights, xx0, xx1, xx2, xx3, dp0, dp1, dp1, dp1, vertIndices, restDist);
    }
}

// ===========================================================================
// cloth_queryClothClothContactEECountLaunch (FEMClothExternalSolve.cu:964)
// ===========================================================================
[shader("compute")]
[numthreads(RC_BLOCK, 1, 1)] // SB_UPDATEROTATION
void cloth_queryClothClothContactEECountLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gQuery.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    Ptr<PxgFEMClothGpu> cloths = Ptr<PxgFEMClothGpu>(gQuery.cloths);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + groupThreadId.x + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint64_t ciBase = gQuery.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint aux = ciAuxInd(ciBase);
        if (!auxIsValidPair(aux))
        {
            ciWriteAuxInd(ciBase, auxMarkInCollision(aux, false));
            continue;
        }

        const uint pairInd0 = ciPairInd0Lo(ciBase);
        const uint pairInd1 = ciPairInd1(ciBase);
        const uint e0_localIndex0 = auxGetInd0(aux);
        const uint e1_localIndex0 = auxGetInd1(aux);
        const uint clothId0 = pxGetClothId(pairInd0);
        const uint clothId1 = pxGetClothId(pairInd1);
        const uint elementId0 = pxGetClothElementIndex(pairInd0);
        const uint elementId1 = pxGetClothElementIndex(pairInd1);

        const uint e0_localIndex1 = (e0_localIndex0 + 1u) % 3u;
        const uint4 triVertInd0 = Ptr<uint4>(cloths[clothId0].mTriangleVertexIndices)[elementId0];
        const uint vi0 = pick3(triVertInd0, e0_localIndex0);
        const uint vi1 = pick3(triVertInd0, e0_localIndex1);

        const uint e1_localIndex1 = (e1_localIndex0 + 1u) % 3u;
        const uint4 triVertInd1 = Ptr<uint4>(cloths[clothId1].mTriangleVertexIndices)[elementId1];
        const uint vi2 = pick3(triVertInd1, e1_localIndex0);
        const uint vi3 = pick3(triVertInd1, e1_localIndex1);
        const uint4 vertIndices = uint4(vi0, vi1, vi2, vi3);

        const float3 xx0 = Ptr<float4>(cloths[clothId0].mPosition_InvMass)[vi0].xyz;
        const float3 xx1 = Ptr<float4>(cloths[clothId0].mPosition_InvMass)[vi1].xyz;
        const float3 xx2 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vi2].xyz;
        const float3 xx3 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vi3].xyz;

        float s, t, distSq;
        closestPtLineLine(xx0, xx1, xx2, xx3, s, t, distSq);
        const bool isOutside = (s < DEFORMABLE_BARYCENTRIC_THRESHOLD) || (s > DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD) ||
                               (t < DEFORMABLE_BARYCENTRIC_THRESHOLD) || (t > DEFORMABLE_ONE_MINUS_BARYCENTRIC_THRESHOLD);
        if (isOutside)
        {
            ciWriteAuxInd(ciBase, auxMarkInCollision(aux, false));
            continue;
        }

        const float4 weights = float4(1.0f - s, s, -(1.0f - t), -t);   // cloth0 - cloth1
        const float restDist = cloths[clothId0].mRestDistance + cloths[clothId1].mRestDistance;

        // countContact: cloth0or1 == cloth0 (ep1 belongs to cloth0).
        const uint64_t dp0 = cloths[clothId0].mDeltaPos;
        const uint64_t dp1 = cloths[clothId1].mDeltaPos;
        countContact(ciBase, aux, weights, xx0, xx1, xx2, xx3, dp0, dp0, dp1, dp1, vertIndices, restDist);
    }
}

// ===========================================================================
// cloth_solveClothClothDeltaVTLaunch (FEMClothExternalSolve.cu:651)
// ===========================================================================
[shader("compute")]
[numthreads(RC_BLOCK, 1, 1)] // __launch_bounds__(256, 4)
void cloth_solveClothClothDeltaVTLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSolve.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    Ptr<PxgFEMClothGpu> cloths = Ptr<PxgFEMClothGpu>(gSolve.cloths);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + groupThreadId.x + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint64_t ciBase = gSolve.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint aux = ciAuxInd(ciBase);
        if (!auxIsInCollision(aux))
            continue;

        const uint pairInd0 = ciPairInd0Lo(ciBase);
        const uint pairInd1 = ciPairInd1(ciBase);
        const uint clothId0 = pxGetClothId(pairInd0);
        const uint clothId1 = pxGetClothId(pairInd1);
        const uint elementId0 = pxGetClothElementIndex(pairInd0);
        const uint elementId1 = pxGetClothElementIndex(pairInd1);

        const uint4 triVertId1 = Ptr<uint4>(cloths[clothId1].mTriangleVertexIndices)[elementId1];
        const uint4 vertIndices = uint4(elementId0, triVertId1.x, triVertId1.y, triVertId1.z);

        const float4 x0 = Ptr<float4>(cloths[clothId0].mPosition_InvMass)[vertIndices.x];
        const float4 x1 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vertIndices.y];
        const float4 x2 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vertIndices.z];
        const float4 x3 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vertIndices.w];

        const float4 prevX0 = Ptr<float4>(cloths[clothId0].mPrevPositionInContactOffset)[vertIndices.x];
        const float4 prevX1 = Ptr<float4>(cloths[clothId1].mPrevPositionInContactOffset)[vertIndices.y];
        const float4 prevX2 = Ptr<float4>(cloths[clothId1].mPrevPositionInContactOffset)[vertIndices.z];
        const float4 prevX3 = Ptr<float4>(cloths[clothId1].mPrevPositionInContactOffset)[vertIndices.w];

        const float4 invMasses = float4(x0.w, x1.w, x2.w, x3.w);
        const float3 xx0 = x0.xyz;
        const float3 xx1 = x1.xyz;
        const float3 xx2 = x2.xyz;
        const float3 xx3 = x3.xyz;

        float s, t;
        const float3 closestPt = closestPtPointTriangle(xx0, xx1, xx2, xx3, s, t);
        const float r = 1.0f - s - t;
        const float4 weights = float4(1.0f, -r, -s, -t);   // cloth0 - cloth1

        const float3 disp = weights.x * xx0 + weights.y * xx1 + weights.z * xx2 + weights.w * xx3;
        const float dispSq = dot(disp, disp);

        float3 normal;
        if (dispSq < FEMCLOTH_THRESHOLD)
        {
            // Vertex is nearly on the triangle, use triangle normal.
            const float3 ab = xx2 - xx1;
            const float3 ac = xx3 - xx1;
            normal = cross(ab, ac);

            const float nLenSq = dot(normal, normal);
            if (nLenSq < FEMCLOTH_THRESHOLD)
                continue;   // Degenerate triangle.

            normal = normal * (1.0f / sqrt(nLenSq));

            const float3 toVertex = xx0 - xx1;
            if (dot(normal, toVertex) < 0.0f)
                normal = -normal;
        }
        else
        {
            const float dist = sqrt(dispSq);
            // P9.36: FEMClothExternalSolve.cu:748 -- PxVec3::operator/, which is ONE
            // reciprocal and three multiplies.  The sibling branch at :846 spells the very
            // same operation `disp * (1.0f / dist)`; they are not two lowerings.
            normal = disp * (1.0f / dist);
        }

        const float4 relDelta4 = weights.x * (x0 - prevX0) + weights.y * (x1 - prevX1) +
                                 weights.z * (x2 - prevX2) + weights.w * (x3 - prevX3);
        const float3 relDelta = float3(relDelta4.x, relDelta4.y, relDelta4.z);

        const float restDist = cloths[clothId0].mRestDistance + cloths[clothId1].mRestDistance;

        // applyContactDelta: cloth0or1 == cloth1.
        const uint64_t dp0 = cloths[clothId0].mDeltaPos;
        const uint64_t dp1 = cloths[clothId1].mDeltaPos;
        const uint64_t df0 = cloths[clothId0].mDynamicFrictions;
        const uint64_t df1 = cloths[clothId1].mDynamicFrictions;
        applyContactDelta(dp0, dp1, dp1, dp1, df0, df1, df1, df1, vertIndices, disp, normal,
                          invMasses, weights, relDelta, restDist);
    }
}

// ===========================================================================
// cloth_solveClothClothDeltaEELaunch (FEMClothExternalSolve.cu:760)
// ===========================================================================
[shader("compute")]
[numthreads(RC_BLOCK, 1, 1)] // __launch_bounds__(256, 4)
void cloth_solveClothClothDeltaEELaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint tNumContacts = Ptr<uint>(gSolve.numContacts)[0];
    const uint nbBlocksRequired = (tNumContacts + RC_BLOCK - 1u) / RC_BLOCK;
    const uint nbIterationsPerBlock = (nbBlocksRequired + RC_GRID - 1u) / RC_GRID;

    Ptr<PxgFEMClothGpu> cloths = Ptr<PxgFEMClothGpu>(gSolve.cloths);

    for (uint i = 0u; i < nbIterationsPerBlock; ++i)
    {
        const uint workIndex = i * RC_BLOCK + groupThreadId.x + nbIterationsPerBlock * groupId.x * RC_BLOCK;
        if (workIndex >= tNumContacts)
            return;

        const uint64_t ciBase = gSolve.contactInfos + uint64_t(workIndex) * CI_STRIDE;
        const uint aux = ciAuxInd(ciBase);
        if (!auxIsInCollision(aux))
            continue;

        const uint pairInd0 = ciPairInd0Lo(ciBase);
        const uint pairInd1 = ciPairInd1(ciBase);
        const uint e0_localIndex0 = auxGetInd0(aux);
        const uint e1_localIndex0 = auxGetInd1(aux);
        const uint clothId0 = pxGetClothId(pairInd0);
        const uint clothId1 = pxGetClothId(pairInd1);
        const uint elementId0 = pxGetClothElementIndex(pairInd0);
        const uint elementId1 = pxGetClothElementIndex(pairInd1);

        const uint e0_localIndex1 = (e0_localIndex0 + 1u) % 3u;
        const uint4 triVertInd0 = Ptr<uint4>(cloths[clothId0].mTriangleVertexIndices)[elementId0];
        const uint vi0 = pick3(triVertInd0, e0_localIndex0);
        const uint vi1 = pick3(triVertInd0, e0_localIndex1);

        const uint e1_localIndex1 = (e1_localIndex0 + 1u) % 3u;
        const uint4 triVertInd1 = Ptr<uint4>(cloths[clothId1].mTriangleVertexIndices)[elementId1];
        const uint vi2 = pick3(triVertInd1, e1_localIndex0);
        const uint vi3 = pick3(triVertInd1, e1_localIndex1);
        const uint4 vertIndices = uint4(vi0, vi1, vi2, vi3);

        const float4 x0 = Ptr<float4>(cloths[clothId0].mPosition_InvMass)[vi0];
        const float4 x1 = Ptr<float4>(cloths[clothId0].mPosition_InvMass)[vi1];
        const float4 x2 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vi2];
        const float4 x3 = Ptr<float4>(cloths[clothId1].mPosition_InvMass)[vi3];

        const float4 prevX0 = Ptr<float4>(cloths[clothId0].mPrevPositionInContactOffset)[vi0];
        const float4 prevX1 = Ptr<float4>(cloths[clothId0].mPrevPositionInContactOffset)[vi1];
        const float4 prevX2 = Ptr<float4>(cloths[clothId1].mPrevPositionInContactOffset)[vi2];
        const float4 prevX3 = Ptr<float4>(cloths[clothId1].mPrevPositionInContactOffset)[vi3];

        const float4 invMasses = float4(x0.w, x1.w, x2.w, x3.w);
        const float3 xx0 = x0.xyz;
        const float3 xx1 = x1.xyz;
        const float3 xx2 = x2.xyz;
        const float3 xx3 = x3.xyz;

        float s, t, distSq;
        const bool isValid = closestPtLineLine(xx0, xx1, xx2, xx3, s, t, distSq);

        float4 weights = float4(1.0f - s, s, -(1.0f - t), -t);   // cloth0 - cloth1
        const float3 disp = weights.x * xx0 + weights.y * xx1 + weights.z * xx2 + weights.w * xx3;
        const float dist = sqrt(distSq);

        const float3 normal = disp * (1.0f / dist);

        if (!isValid)   // Degenerate: apply at mid-points.
            weights = float4(0.5f, 0.5f, -0.5f, -0.5f);

        const float4 relDelta4 = weights.x * (x0 - prevX0) + weights.y * (x1 - prevX1) +
                                 weights.z * (x2 - prevX2) + weights.w * (x3 - prevX3);
        const float3 relDelta = float3(relDelta4.x, relDelta4.y, relDelta4.z);

        const float restDist = cloths[clothId0].mRestDistance + cloths[clothId1].mRestDistance;

        // applyContactDelta: cloth0or1 == cloth0.
        const uint64_t dp0 = cloths[clothId0].mDeltaPos;
        const uint64_t dp1 = cloths[clothId1].mDeltaPos;
        const uint64_t df0 = cloths[clothId0].mDynamicFrictions;
        const uint64_t df1 = cloths[clothId1].mDynamicFrictions;
        applyContactDelta(dp0, dp0, dp1, dp1, df0, df0, df1, df1, vertIndices, disp, normal,
                          invMasses, weights, relDelta, restDist);
    }
}

// ===========================================================================
// cloth_updateClothContactValidityLaunch (FEMClothExternalSolve.cu:1114)
// ===========================================================================
[shader("compute")]
[numthreads(CLOTH_STEP, 1, 1)]
void cloth_updateClothContactValidityLaunch(uint3 groupThreadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    const uint id = Ptr<uint>(gUpdate.activeIds)[groupId.y];
    Ptr<PxgFEMClothGpu> femCloths = Ptr<PxgFEMClothGpu>(gUpdate.femCloths);
    const uint nbVerts = femCloths[id].mNbVerts;

    const uint64_t curPositions = femCloths[id].mPosition_InvMass;
    const uint64_t prevPositionsInContactOffset = femCloths[id].mPrevPositionInContactOffset;

    const uint globalThreadIndex = groupThreadId.x + CLOTH_STEP * groupId.x;

    const bool adaptive = (gUpdate.adaptiveFlagWord & 0xFFu) != 0u;
    const bool updateAdaptiveCollisionPair = adaptive && ((Ptr<uint>(gUpdate.updateContactPairs)[0] & 0xFFu) == 1u);

    if (updateAdaptiveCollisionPair && globalThreadIndex == 0u && groupId.y == 0u)
    {
        // Reset the contact count.
        Ptr<uint>(gUpdate.VTContactCount)[0] = 0u;
        Ptr<uint>(gUpdate.EEContactCount)[0] = 0u;
    }

    if (globalThreadIndex < nbVerts)
    {
        float4 curPos = Ptr<float4>(curPositions)[globalThreadIndex];
        if (updateAdaptiveCollisionPair)
        {
            curPos.w = 1.0f;   // Mark prevPosInContactOffset as set.
            Ptr<float4>(prevPositionsInContactOffset)[globalThreadIndex] = curPos;
        }
    }
}
