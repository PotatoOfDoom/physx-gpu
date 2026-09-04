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

// pxgSlangCvxHfSetup.slang -- the convex-vs-HEIGHTFIELD per-pair prologue
// (convexHeightfield.cu:120-269), the heightfield sibling of `ctSetup`.  P9.51b.
//
// WHY THIS IS A MODULE AND NOT A FUNCTION INSIDE THE CORE
//
// The core kernel `convexHeightfieldNarrowphase` has no unit fixture, and neither
// does its trimesh sibling: both are compositions of modules that were each
// verified on their own, and their composition is evidenced by a scene.  That
// works only as long as every part of the composition IS one of those verified
// modules.  The triangle fetch below was the one new part, so it lives here, where
// `cvxHfSetupBda` can drive it against a host mirror -- rather than in the core,
// where the only way to reach it would be to test a copy of it.  P9.15d is the
// precedent that made this rule: two kernels compiling the "same" body drifted
// apart, and the differential that was supposed to catch it measured the copies.
//
// WHY IT IS NOT IN pxgSlangCvxTriSetup
//
// That module states, and depends on, having NO imports -- it is pure address
// arithmetic over two blobs.  This one needs `pxgSlangHeightfield`'s sample
// decoders, and folding it in would pull the heightfield decode into
// `cvxTriPostProcessBda` and every other trimesh consumer that has no use for it.
// The shared half they DO have in common is `ctCommon`, which is what the P9.51b
// split extracted.
//
// ===========================================================================
// THE SLOT SWAP IS ABSENT ON PURPOSE
// ===========================================================================
//
// convexMesh.cu:135 tests slot 0 for eTRIANGLEMESH and swaps the operands when it
// hits.  convexHeightfield.cu does NOT: it reads `gpuShapes[shapeRef0]` as the
// convex and `gpuShapes[shapeRef1]` as the heightfield unconditionally (:211,
// :214).  So this passes CT_NO_SWAP, a value no PxGeometryType can carry.
//
// Leaving the trimesh swap in would be invisible in every scene the current
// broadphase produces -- it emits these pairs convex-first -- and would exchange
// the two operands the first time one did not.  That is the same mutant as
// `H1-trimesh-swap-added` in the P9.51a midphase sweep, and it SURVIVES there for
// exactly this reason: no fixture can present the input that would expose it,
// because upstream never presents it either.  A survivor whose cause is understood
// is a documented equivalence, not an untested branch.

import pxgSlangCvxTriSetup;
import pxgSlangHeightfield;

// The same shape `ctSetup` returns, so the two cores read alike.  There is no
// `remapCpuTriangleIdx` field: a heightfield has no cooked faceRemap, and the .cu
// passes `triangleIdx` for both parameters of convexTriangleContactGen (:266,
// with the comment "height field don't have cpu remap triangle index").  Carrying
// a field that is always a copy of another invites a caller to believe it means
// something.
public struct ChSetup
{
    public uint64_t hull;
    public float3 convexScale;
    public float4 convexScaleRot;
    public float contactDist;
    public float3 triLocVerts0, triLocVerts1, triLocVerts2;
    public float3 triangleLocNormal;
    public uint3 triAdjTrisIdx;
    public uint triangleIdx;
    public uint convexTriPairOffset;
    public uint convexTriPairOffsetPadded;
}

// buildTriangleInformations (convexHeightfield.cu:120-157).
//
// The heightfield blob is `[0] rows, [1] columns, then the samples`, so the
// samples begin at meshPtr + 8 (:123-127).
//
// THE NORMAL IS ROTATED BEFORE IT IS NORMALIZED, AND FROM THE DENORMALIZED CROSS
// (:136-137): `t2c.rotate(triLocal.denormalizedNormal()).getNormalized()`.
// Normalizing first gives the same direction in exact arithmetic and a different
// last bit in float -- and this normal is the SAT's reference axis, so a last bit
// here moves `minNormal` and can move which polygon feature gets selected.  The
// trimesh sibling spells it the same way for the same reason.
//
// `rotate` is the rotation ALONE, with no translation (:137); the VERTICES get the
// full transform (:155).  Writing `transform` for the normal would add the
// translation to a direction, which is wrong by a whole convex-to-mesh offset
// rather than by a bit -- and would still leave a unit vector after the
// normalize, so the output would look well-formed.
public ChSetup chSetup(uint64_t cmInputs, uint64_t transformCache, uint64_t contactDistance,
                       uint64_t gpuShapes, uint64_t cvxTrimeshPair, uint4 curPair)
{
    const CtCommon c = ctCommon(cmInputs, transformCache, contactDistance, gpuShapes,
                                cvxTrimeshPair, curPair, CT_NO_SWAP);

    const uint rows = Ptr<uint>(c.meshPtr)[0];
    const uint columns = Ptr<uint>(c.meshPtr + 4UL)[0];
    const uint64_t samples = c.meshPtr + 8UL;

    const HfTriangle t = hfGetTriangleAdj(samples, c.triangleIdx, c.meshScale, rows, columns);

    ChSetup s;
    s.hull = c.hull;
    s.convexScale = c.convexScale;
    s.convexScaleRot = c.convexScaleRot;
    s.contactDist = c.contactDist;
    s.triangleIdx = c.triangleIdx;
    s.convexTriPairOffset = c.convexTriPairOffset;
    s.convexTriPairOffsetPadded = c.convexTriPairOffsetPadded;

    s.triAdjTrisIdx = t.adj;
    s.triangleLocNormal = ctNormalized(ctQuatRotate(c.toCvxQ, cross(t.v1 - t.v0, t.v2 - t.v0)));
    s.triLocVerts0 = ctQuatRotate(c.toCvxQ, t.v0) + c.toCvxP;
    s.triLocVerts1 = ctQuatRotate(c.toCvxQ, t.v1) + c.toCvxP;
    s.triLocVerts2 = ctQuatRotate(c.toCvxQ, t.v2) + c.toCvxP;

    // THE HEIGHTFIELD'S SCALE IS NOT THE CONVEX'S.  `c.meshScale` is slot 1's
    // PxMeshScale.scale, which for a heightfield is (rowScale, heightScale,
    // columnScale) -- the units-to-shape-space factors hf2shapep applies (:114).
    // `c.convexScale` above is slot 0's and drives the hull.  They are two
    // different three-vectors that both arrive as "the scale".
    return s;
}
