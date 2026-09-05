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

// pxgSlangSphereHfCore.slang -- `sphereHeightfieldNarrowphaseCore`
// (convexHeightfield.cu:359-562) on the engine BDA ABI.  P9.52a.
//
// This is the SPHERE arm of `sphereHeightfieldNarrowphase`, the one kernel the
// sphere-vs-heightfield chain is still missing.  The other five stages of that
// chain are already mapped and green -- CONVEX_HEIGHTFIELD_MIDPHASE (P9.51a),
// CONVEX_TRIMESH_SORT_TRIANGLES (P8.48), CONVEX_HEIGHTFIELD_POST_PROCESS (P9.51c,
// the same kernel id serves both paths), CONVEX_TRIMESH_CORRELATE (P8.50b) and
// CONVEX_TRIMESH_FINISHCONTACTS (P8.51) -- so this is the only hole left in it.
//
// ===========================================================================
// THE MANIFEST ROW STAYS `nullptr` UNTIL THE CAPSULE ARM LANDS TOO
// ===========================================================================
//
// `sphereHeightfieldNarrowphase` dispatches on slot 0's geometry type
// (convexHeightfield.cu:886-928): eSPHERE reaches THIS core, eCAPSULE reaches
// `capsuleHeightfieldNarrowphaseCore` (:636-789), which needs
// `distanceSegmentTriangleSquared`, `computeTriangleFlags`, `selectNormal` and the
// two capsuleTriangle.cuh contact generators -- none of which exist on the BDA ABI
// yet.  Mapping the kernel with only this arm would produce a kernel that handles
// spheres and drops capsules IN SILENCE, which is exactly the P9.27/P9.40 defect
// this port spent a tranche eliminating.  Unmapped, `[pxg-unmapped]` still tells
// the truth about the whole kernel; half-mapped, nothing would.
//
// ===========================================================================
// WHY THIS IS A MODULE WITH A FIXTURE, AND NOT A FUNCTION IN THE KERNEL
// ===========================================================================
//
// Same rule as `pxgSlangCvxHfSetup` (P9.51b): a kernel with no unit fixture is
// only as trustworthy as the modules it is composed of, so the parts that are NEW
// go in a module a fixture can drive against a host mirror.  Everything below is
// new -- the sphere prologue, the edge-concavity switch and the record tail are
// none of them shared with the convex path.
//
// ===========================================================================
// WHAT IT DOES NOT SHARE WITH THE CONVEX PATH, DESPITE LOOKING LIKE IT
// ===========================================================================
//
//  1. THE TRIANGLE NORMAL IS NORMALIZED IN HEIGHTFIELD SPACE, NOT AFTER A
//     ROTATION.  `chSetup` spells it `normalize(rotate(toCvxQ, denormalizedCross))`
//     because convexHeightfield.cu:136-137 does; this core spells it
//     `normalize(denormalizedCross)` because :412 does.  There is no rotation to
//     apply: everything here happens in the HEIGHTFIELD's local frame -- it is the
//     SPHERE CENTRE that moves, not the triangle.
//
//  2. THE TRANSFORM COMPOSES THE OTHER WAY ROUND.  `ctCommon` builds
//     convex-relative (`conj(q0)*q1`, `rotInv(q0, p1 - p0)`) because the convex
//     path drags the triangle into the hull's frame.  This core forms only
//     `sphereCenterInTriMesh = trimeshTransform.transformInv(sphereTransform.p)`
//     = `rotInv(q1, p0 - p1)` (:435) -- mesh-relative, and a POINT rather than a
//     transform.  Reusing `ctCommon` here would compute a quaternion product the
//     CUDA never forms and hand back a translation with the wrong sign convention.
//     That is why this prologue is its own, and why only the DECODERS (the shape
//     and contact-manager layouts, the quaternion helpers) come from that module:
//     those are the parts where a defect would otherwise have to be fixed twice.
//
//  3. THE RADIUS IS `scale.scale.X` (:428).  The OUTPUT stage of the same chain
//     reads `scale.scale.Y` for the same sphere (convexMeshOutput.cu:314).  Both
//     are correct and the difference is not a bug: `PxgNarrowphaseCore.cpp:7508`
//     fills a sphere's scale as `PxVec3(sphere.radius)` -- all three components
//     equal -- while a CAPSULE gets `PxVec3(halfHeight, radius, radius)`, and the
//     output stage is shared between the two, so it must read `.y`.  This core
//     only ever sees spheres, so `.x` is as good as `.y` THERE and not here.
//     The fixture drives an anisotropic sphere scale anyway; see its header for
//     why a synthetic input is the honest way to pin which component is read.
//
//  4. THE ORDERED WORD HAS ONLY THE FACE ARM.  `cteEmitTail` writes
//     `faceFlag | triangleIdx` where faceFlag also covers an eEDGE_CONTACT case;
//     this core writes `(nb && !delay) ? (0x80000000 | tri) : tri` (:548) and has
//     no edge arm at all.  The postprocess reads bit 31 as "this pair produced a
//     face contact", and adding an arm here would make grid-edge duplicates
//     survive.
//
//  5. `index` IS AN ADDITION, NOT AN OR (:556).  The three fields are disjoint, so
//     the two spellings agree on every input the engine produces; it is written as
//     the CUDA writes it because a transcription that "cleans up" an arithmetic op
//     is a transcription that stopped being checkable.
//
// ===========================================================================
// THE EDGE-CONCAVITY SWITCH, AND THE ONE PLACE THE PORT DIVERGES ON PURPOSE
// ===========================================================================
//
// `ptDistancePointTriangleSquared`'s own convexity test is INERT on a heightfield:
// `isEdgeNonconvex` reads bit 31 of an adjacency index, and heightfield adjacency
// indices are raw triangle indices produced by grid arithmetic, so bit 31 is only
// ever set by the BOUNDARY sentinel 0xFFFFFFFF -- which the switch below filters
// first.  (P9.55 measured the other half of the same fact: `edgeTol`, the
// tie-breaker in the same branches, is unreachable here for the same reason.)
//
// The real test is the switch at :466-516: for an eE01 / eE12 / eE02 closest
// feature the ADJACENT triangle is re-fetched and the contact survives only if the
// current triangle's OPPOSITE vertex lies below the adjacent face's plane.  The
// three arms pair different vertices with different adjacency slots, and getting
// the pairing wrong is invisible on a flat heightfield:
//
//     eE01 -> adj.x, tests triLocV2   (:474)
//     eE12 -> adj.y, tests triLocV0   (:488)
//     eE02 -> adj.z, tests triLocV1   (:503)
//
// **THE BOUNDARY GUARD IS NOT IN THE CUDA.**  Upstream calls `getTriangle` on the
// adjacency index without testing it against 0xFFFFFFFF, so a sphere resting on an
// edge of the grid's rim makes the switch decode triangle 0xFFFFFFFF: cell
// 0x7FFFFFFF, sample indices far past the end of the blob.  Nothing faults on a
// GPU with a large enough mapped heap; `generateContact` is then decided by
// whatever was in that memory.  A Vulkan buffer_reference read at that offset is
// undefined behaviour with no such luck.
//
// The port therefore SKIPS the switch on a BOUNDARY adjacency, which keeps the
// contact -- the reading upstream would have taken if the garbage happened to be
// negative, and the one the geometry argues for: a rim edge has no neighbouring
// face to occlude it, so there is nothing to reject.
//
// AND IT IS NOT THE PORT'S INVENTION.  Twenty lines above this core, in the SAME
// FILE and over the SAME adjacency array, `computeTriangleFlags` (:583) writes
// `if (adjTriIndices[a] != 0xFFFFFFFF)` -- the exact test the sphere arm omits.
// The convention is upstream's; only this one arm does not follow it.  That is
// UPSTREAM CANDIDATE 49, recorded in RUNTIME_STATUS with the twin as its evidence:
// DOCUMENTED, NOT REPORTED, like the other 49.
//
// The sweep measures what the unguarded form costs here: `C10-boundary-guard-
// dropped` is the upstream spelling, and it segfaults 4 times out of 4.
//
// ===========================================================================
// `triAdjTriIndices.w` IS UNINITIALISED UPSTREAM
// ===========================================================================
//
// getTriangle's uint4 out-parameter assigns .x, .y and .z only
// (heightfieldUtil.cuh:295-297).  `ptDistancePointTriangleSquared` reads .x, .y
// and .z, never .w -- which is what makes the CUDA well-defined.  The uint4 handed
// over below carries a zero in .w so the port has no undefined lane at all, and
// nothing may start reading it.

// ===========================================================================
// THE MUTATION SWEEP: 64 MUTANTS, 57 KILLED, AND SEVEN EQUIVALENCES
// ===========================================================================
//
// 52 killed by a failing fixture, 5 KILLED BY FAULT, 7 survived.  The five faults
// are themselves evidence and not noise: `A3-heightfield-from-slot0`,
// `C1-eE01-uses-adj-y`, `C3-eE12-fetches-adj-z`, `C10-boundary-guard-dropped` and
// `C11-eE02-guard-tests-adj-x` all end up decoding a BOUNDARY adjacency or reading
// a sphere's null blob as a sample grid, and under Vulkan that is a segfault rather
// than the plausible garbage the same code yields in CUDA.  That is exactly what
// the guard above exists to prevent.
//
// EVERY survivor is an EQUIVALENCE, not an untested branch, and they fall into
// four families.  (The first sweep left ELEVEN; four of those were genuine fixture
// gaps -- the E12 arm never answered `keep`, no case had one adjacency slot as the
// sentinel and another real, and no accepted contact had a cosTheta between the two
// extremes -- and the fixture now carries a named guard for each.)
//
//  (a) EXACT-ARITHMETIC EQUIVALENTS -- the two spellings compute the same real
//      number and differ only by float rounding of a quantity that is identically
//      zero:
//        B4-dist-measured-from-v1   `dot(centre - v0, n)` vs `dot(centre - v1, n)`.
//                                   v0 and v1 both lie in the plane whose normal is
//                                   n, so the difference is `dot(n, v1 - v0)` == 0.
//        C8-eE01-plane-point-is-a1  `v2 - a0` vs `v2 - a1` against the NEIGHBOUR's
//                                   normal.  a0 and a1 are both its vertices, so
//                                   `dot(nAdj, a1 - a0)` == 0 by construction.
//      Neither can be killed by any input a heightfield can present: the residue is
//      ~1 ulp while the decision value is ~1e-2 wherever the geometry is not
//      exactly coplanar, and where it IS exactly coplanar (integer heights, a
//      power-of-two height scale) BOTH spellings give exactly zero.
//
//  (b) A POSITIVE SCALING OF A SIGN TEST:
//        C12-adj-normal-not-normalized   the neighbour's normal feeds only
//        `dot(nAdj, x) < 0`, and normalizing multiplies by a positive scalar.  The
//        degenerate case agrees too: getNormalized maps the zero vector to the zero
//        vector, which is what the raw cross already was.  The port keeps the
//        normalize because the .cu has it.
//
//  (c) DISJOINT BIT FIELDS -- `+` and `|` agree because the operands never overlap:
//        D11-ni-index-is-or     deferred is bit 31, the count lands on bit 27, and
//                               a triangle index is below 2^27 (the field the
//                               engine's own getTriangleIndex masks out).
//        E9-outmask-or-is-add   the region mask occupies bits 29-31; a pair index
//                               that reached bit 29 would mean 2^29 contact-manager
//                               records, which is more than any device can hold.
//      Upstream writes `+` in one place and `|` in the other, and so does this
//      module -- transcribed, not tidied.
//
//  (d) PROVABLY DEAD INPUTS:
//        B3-adj-w-is-boundary        `ptDistancePointTriangleSquared` reads .x, .y
//                                    and .z and never .w (pxgSlangPointTriangle:
//                                    198-230), which is also what makes upstream's
//                                    uninitialised .w well-defined.
//        B7-generate-contact-ignored the primitive's own convexity verdict is
//                                    IDENTICALLY TRUE on a heightfield:
//                                    `isEdgeNonconvex` is false for every raw grid
//                                    index and false for the BOUNDARY sentinel it
//                                    excludes by name, so each vertex branch's
//                                    disjunction and each edge branch's
//                                    short-circuit are true, and the face branch
//                                    assigns true outright.  This survivor is the
//                                    measurement behind the claim made above that
//                                    the switch, not the primitive, decides edge
//                                    convexity here.

module pxgSlangSphereHfCore;

import pxgSlangCvxTriSetup;      // CtShape / CtContactManagerInput / the quat helpers
import pxgSlangHeightfield;      // the sample decoders and both getTriangle arms
import pxgSlangPointTriangle;    // sphereTriangle.cuh:41, one copy (P9.56)

// PX_MAX_F32 -- what the CUDA seeds `separation` with (:441) and therefore what
// lands in sphereTriMaxDepth for a pair that generates nothing.
static const float SH_MAX_F32 = 3.4028234663852886e+38f;
// ConvexTriNormalAndIndex (convexNpCommon.h:84-85), via the SphereTri typedef
// at :142 -- the same struct under a second name.
public static const uint SH_DEFERRED_CONTACT_MASK = 0x80000000u;
public static const uint SH_NB_CONTACTS_SHIFT = 27u;
// The `(nbOutputContacts && !delayContacts)` stamp on the ordered index (:548).
static const uint SH_FACE_CONTACT_FLAG = 0x80000000u;
// "around 5 degree" (:530).
static const float SH_COS_TOLERANCE = 0.996f;

// What the core hands back to its caller: the CUDA's return value plus its three
// out-parameters, plus the pair offset the kernel's output stage needs for the
// ConvexTriContacts linkage (:986).
public struct ShCoreOut
{
    public uint nbOutputContacts;
    public uint outMask;
    // ZERO here where the CUDA leaves the caller's `shContacts` slot untouched on
    // the no-contact path.  It is never read there -- the output loop runs
    // `i < nbContacts` -- so a defined zero is the same behaviour with one less
    // uninitialised register.
    public float3 outContact;
    public float outSep;
    public uint pairOffset;
}

// `globalThreadIndex` is the CUDA's pair index within the whole dispatch, and it
// goes into the deferred-pair record verbatim (:534): the second pass re-derives
// the pair from it, so it must be the same numbering the kernel loop used.
//
// `ni`, `maxDepth`, `interm` and `ordered` are the ALREADY-RESOLVED bases the
// CUDA reaches through `*sphereTriNIPtr` and friends -- in the engine kernel they
// come from `maCompute`, in the fixture from plain buffers.
public ShCoreOut shSphereHfCore(uint globalThreadIndex,
                                uint64_t cmInputs, uint64_t transformCache,
                                uint64_t contactDistance, uint64_t gpuShapes,
                                uint64_t sphereTrimeshPair,
                                uint64_t ni, uint64_t maxDepth, uint64_t interm,
                                uint64_t ordered,
                                uint4 curPair)
{
    const uint cmIdx = curPair.x;
    const uint triangleIdx = curPair.y;
    const uint testOffset = curPair.z;

    const CtContactManagerInput cin = Ptr<CtContactManagerInput>(cmInputs)[cmIdx];

    // NO SWAP, and no test for one: :392 and :396 read slot 0 as the sphere and
    // slot 1 as the heightfield unconditionally, exactly as the convex-vs-
    // heightfield path does and unlike convexMesh.cu:135.
    Ptr<CtShape> shapes = Ptr<CtShape>(gpuShapes);
    const CtShape sphereShape = shapes[cin.shapeRef0];
    const CtShape heightfieldShape = shapes[cin.shapeRef1];

    // PxsCachedTransform, 32 B: PxQuat q @0, PxVec3 p @16.
    const uint64_t xf0 = transformCache + uint64_t(cin.transformCacheRef0) * 32UL;
    const uint64_t xf1 = transformCache + uint64_t(cin.transformCacheRef1) * 32UL;
    const float4 sphereQ = Ptr<float4>(xf0)[0];
    const float3 sphereP = float3(Ptr<float>(xf0 + 16UL)[0], Ptr<float>(xf0 + 20UL)[0],
                                  Ptr<float>(xf0 + 24UL)[0]);
    const float4 meshQ = Ptr<float4>(xf1)[0];
    const float3 meshP = float3(Ptr<float>(xf1 + 16UL)[0], Ptr<float>(xf1 + 20UL)[0],
                                Ptr<float>(xf1 + 24UL)[0]);

    // The heightfield blob: [0] rows, [1] columns, then the samples (:398-401).
    const uint64_t meshPtr = uint64_t(heightfieldShape.hullOrMeshOffset.x)
                           | (uint64_t(heightfieldShape.hullOrMeshOffset.y) << 32);
    const uint rows = Ptr<uint>(meshPtr)[0];
    const uint columns = Ptr<uint>(meshPtr + 4UL)[0];
    const uint64_t samples = meshPtr + 8UL;

    const float3 hfScale = float3(heightfieldShape.scaleX, heightfieldShape.scaleY,
                                  heightfieldShape.scaleZ);

    const HfTriangle t = hfGetTriangleAdj(samples, triangleIdx, hfScale, rows, columns);
    const uint4 adj = uint4(t.adj, 0u);   // .w is never read -- see the header

    // NOT rotated first -- see trap 1 in the header.
    const float3 triNormal = ptNormalized(ptCross(t.v1 - t.v0, t.v2 - t.v0));

    // ConvexMeshPair, 56 B: startIndex @28, roundedStartIndex @40.  The trailing
    // numbers in that declaration are RUNNING END offsets, not field offsets --
    // reading them as field offsets lands on `count` and `cmIndex`, which are also
    // small non-negative integers, so nothing faults and the records go somewhere
    // plausible-looking.  (The same trap ctCommon documents; it is restated here
    // because this prologue does its own read.)
    const uint64_t pairAddr = sphereTrimeshPair + uint64_t(cmIdx) * 56UL;
    const uint pairOffset = Ptr<uint>(pairAddr + 28UL)[0] + testOffset;
    const uint pairOffsetPadded = Ptr<uint>(pairAddr + 40UL)[0] + testOffset;

    // Indexed by the TRANSFORM refs, not the shape refs (:426).
    const float contactDist = Ptr<float>(contactDistance)[cin.transformCacheRef0]
                            + Ptr<float>(contactDistance)[cin.transformCacheRef1];
    const float sphereRadius = sphereShape.scaleX;    // .x, not .y -- see trap 3

    // transformInv: rotInv(q, v - p).  Mesh-relative -- see trap 2.
    const float3 sphereCenterInTriMesh = ctQuatRotateInv(meshQ, sphereP - meshP);

    const float inflatedRadius = sphereRadius + contactDist;

    // ONE-SIDED (:453, :459): a sphere whose centre is behind the triangle's plane
    // generates nothing at all, however close it is.  The heightfield is a
    // heightfield, not a shell.
    const float dist = ptDot(sphereCenterInTriMesh - t.v0, triNormal);

    float separation = SH_MAX_F32;
    uint nbOutputContacts = 0u;
    bool delayContacts = false;
    float3 patchNormal = float3(0.0f, 0.0f, 0.0f);
    uint outMask = 0u;
    float3 outContact = float3(0.0f, 0.0f, 0.0f);
    float outSep = 0.0f;

    float3 closestP;
    bool generateContact = false;
    bool faceContact = false;
    uint mask;
    const float sqDist = ptDistancePointTriangleSquared(sphereCenterInTriMesh,
                                                        t.v0, t.v1, t.v2, adj,
                                                        closestP, generateContact,
                                                        faceContact, mask);

    const float sqInflatedSphereRadius = inflatedRadius * inflatedRadius;
    if (dist >= 0.0f && sqInflatedSphereRadius > sqDist && generateContact)
    {
        // The edge-concavity switch (:466-516).  The BOUNDARY guards are the
        // port's, not upstream's -- see the header.
        if (mask == PT_EE01 && adj.x != HF_BOUNDARY)
        {
            float3 a0, a1, a2;
            hfGetTriangle(samples, adj.x, hfScale, columns, a0, a1, a2);
            const float3 nAdj = ptNormalized(ptCross(a1 - a0, a2 - a0));
            generateContact = ptDot(nAdj, t.v2 - a0) < 0.0f;
        }
        else if (mask == PT_EE12 && adj.y != HF_BOUNDARY)
        {
            float3 a0, a1, a2;
            hfGetTriangle(samples, adj.y, hfScale, columns, a0, a1, a2);
            const float3 nAdj = ptNormalized(ptCross(a1 - a0, a2 - a0));
            generateContact = ptDot(nAdj, t.v0 - a0) < 0.0f;
        }
        else if (mask == PT_EE02 && adj.z != HF_BOUNDARY)
        {
            float3 a0, a1, a2;
            hfGetTriangle(samples, adj.z, hfScale, columns, a0, a1, a2);
            const float3 nAdj = ptNormalized(ptCross(a1 - a0, a2 - a0));
            generateContact = ptDot(nAdj, t.v1 - a0) < 0.0f;
        }

        if (generateContact)
        {
            patchNormal = triNormal;
            if (!faceContact)
                patchNormal = ptNormalized(sphereCenterInTriMesh - closestP);

            const float cosTheta = ptDot(patchNormal, triNormal);
            delayContacts = cosTheta <= SH_COS_TOLERANCE;
            if (delayContacts)
            {
                // The RAW mask, not a re-derived barycentric classification: the
                // trimesh emit tail recomputes the region from the clipped contact
                // (cteBarycentric + cteTriangleActiveMask) because its contact came
                // out of a clipper; here the region is already known exactly.
                outMask = globalThreadIndex | mask;
            }

            nbOutputContacts = 1u;
            // THE DISTANCE FROM THE CENTRE, not a penetration: the radius is
            // subtracted much later, in the output stage (convexMeshOutput.cu:313).
            separation = sqrt(sqDist);

            // The contact point is the sphere's own origin; the output stage
            // reconstructs the surface point as `p - worldNormal * radius`.
            outContact = float3(0.0f, 0.0f, 0.0f);
            outSep = separation;
        }
    }

    Ptr<float>(maxDepth)[pairOffset] = separation;
    Ptr<uint>(interm)[pairOffset] = triangleIdx;
    // The FACE arm only -- see trap 4.  At the PADDED offset, which is the sort's
    // numbering and not the record numbering.
    Ptr<uint>(ordered)[pairOffsetPadded] =
        (nbOutputContacts != 0u && !delayContacts) ? (SH_FACE_CONTACT_FLAG | triangleIdx)
                                                   : triangleIdx;

    const uint delayContactMask = delayContacts ? SH_DEFERRED_CONTACT_MASK : 0u;
    // Into the SPHERE's frame by way of the world: rotate by the mesh, rotate back
    // by the sphere (:560-561).  Both are rotations alone -- a normal is a
    // direction, and adding either translation would still leave a unit vector, so
    // the mistake would survive every well-formedness check downstream.
    const float3 worldNormal = ctQuatRotate(meshQ, -patchNormal);
    const float3 niNormal = ctQuatRotateInv(sphereQ, worldNormal);
    // A heightfield never remaps: the RAW gpu triangle index goes in (:556).
    const uint niIndex = delayContactMask + (nbOutputContacts << SH_NB_CONTACTS_SHIFT)
                       + triangleIdx;

    Ptr<uint>(ni)[4u * pairOffset + 0u] = asuint(niNormal.x);
    Ptr<uint>(ni)[4u * pairOffset + 1u] = asuint(niNormal.y);
    Ptr<uint>(ni)[4u * pairOffset + 2u] = asuint(niNormal.z);
    Ptr<uint>(ni)[4u * pairOffset + 3u] = niIndex;

    ShCoreOut o;
    o.nbOutputContacts = nbOutputContacts;
    o.outMask = outMask;
    o.outContact = outContact;
    o.outSep = outSep;
    o.pairOffset = pairOffset;
    return o;
}
